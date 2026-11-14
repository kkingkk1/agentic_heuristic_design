import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(directed_valid):
        return np.zeros((n, n), dtype=float)

    # Use the best available direction while retaining only truly valid pairs.
    safe = np.where(directed_valid, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite = safe[valid]
        global_scale = float(np.median(finite)) if finite.size else 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    # Nearest-neighbor local scales. Invalid entries remain at infinity.
    order = np.argsort(safe, axis=1, kind="stable")
    neighbor_count = min(4, n - 1)

    nearest = np.take_along_axis(safe, order[:, :neighbor_count], axis=1)
    finite_nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    local_scale = np.mean(finite_nearest, axis=1)
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    # Short-edge affinity, scaled by the local neighborhoods of both endpoints.
    relative_distance = safe / pair_scale
    short_edge = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    # Endpoint rank support: high only when an edge is locally competitive
    # for both endpoints.
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    col_order = np.argsort(safe, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    rank_support = 0.5 * (
        np.exp(-row_rank / 2.0) + np.exp(-col_rank / 2.0)
    )

    top_k = min(2, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True

    mutual_top = (top_neighbors & top_neighbors.T & valid).astype(float)

    nearest_row = order[:, 0]
    nearest_col = col_order[0, :]
    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Small bounded penalty for edges that are not competitive at either end.
    competitive = np.minimum(
        np.exp(-row_rank / max(float(top_k), 1.0)),
        np.exp(-col_rank / max(float(top_k), 1.0)),
    )
    competitive_penalty = 0.90 + 0.10 * competitive

    scores = (
        0.66 * short_edge
        + 0.24 * rank_support
        + 0.06 * mutual_top
        + 0.04 * mutual_nearest
    )
    scores *= competitive_penalty

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
