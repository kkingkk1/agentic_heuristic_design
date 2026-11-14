import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    valid_input = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid_input):
        return scores

    eps = np.finfo(float).eps
    finite_values = distances[valid_input]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    # Keep only edges supported in at least one direction and make them symmetric.
    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return scores

    safe = safe.copy()
    np.fill_diagonal(safe, np.inf)

    # Adaptive endpoint-local scale from the nearest few usable edges.
    neighborhood_size = min(5, n - 1)
    nearest_values = np.partition(
        safe,
        kth=neighborhood_size - 1,
        axis=1,
    )[:, :neighborhood_size]

    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    normalized_distance = safe / pair_scale
    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    endpoint_row_score = np.exp(
        -np.clip(safe / local_scale[:, None], 0.0, 50.0)
    )
    endpoint_col_score = np.exp(
        -np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )
    endpoint_affinity = np.sqrt(
        endpoint_row_score * endpoint_col_score
    )

    # Obtain stable truncated ranks without sorting every row and column fully.
    rank_window = min(8, n - 1)

    row_candidates = np.argpartition(
        safe,
        kth=rank_window - 1,
        axis=1,
    )[:, :rank_window]
    row_candidate_distances = np.take_along_axis(
        safe,
        row_candidates,
        axis=1,
    )
    row_order = np.argsort(
        row_candidate_distances,
        axis=1,
        kind="stable",
    )
    row_neighbors = np.take_along_axis(
        row_candidates,
        row_order,
        axis=1,
    )

    col_candidates = np.argpartition(
        safe,
        kth=rank_window - 1,
        axis=0,
    )[:rank_window, :]
    col_candidate_distances = np.take_along_axis(
        safe,
        col_candidates,
        axis=0,
    )
    col_order = np.argsort(
        col_candidate_distances,
        axis=0,
        kind="stable",
    )
    col_neighbors = np.take_along_axis(
        col_candidates,
        col_order,
        axis=0,
    )

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

    np.put_along_axis(
        row_rank,
        row_neighbors,
        np.broadcast_to(
            np.arange(rank_window, dtype=float),
            (n, rank_window),
        ),
        axis=1,
    )
    np.put_along_axis(
        col_rank,
        col_neighbors,
        np.broadcast_to(
            np.arange(rank_window, dtype=float)[:, None],
            (rank_window, n),
        ),
        axis=0,
    )

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    nearest_row = row_neighbors[:, 0]
    nearest_col = col_neighbors[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_k = min(2, n - 1)
    row_top = np.zeros((n, n), dtype=bool)
    col_top = np.zeros((n, n), dtype=bool)

    np.put_along_axis(
        row_top,
        row_neighbors[:, :top_k],
        True,
        axis=1,
    )
    np.put_along_axis(
        col_top,
        col_neighbors[:top_k, :],
        True,
        axis=0,
    )

    mutual_top = (row_top & col_top & valid).astype(float)

    # Mildly compensate edges incident to locally isolated endpoints.
    isolation_ratio = pair_scale / (safe + global_scale)
    isolation_ratio = np.clip(
        np.nan_to_num(
            isolation_ratio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_affinity
        * (0.48 + 0.52 * reciprocal_rank)
        * (1.0 + 0.36 * mutual_nearest + 0.12 * mutual_top)
        * (1.0 + 0.18 * isolation_ratio)
    )

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
