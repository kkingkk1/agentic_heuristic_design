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
    valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, np.finfo(float).eps)

    safe = np.where(valid, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    sorted_distances = np.sort(safe, axis=1)
    usable = min(5, n - 1)

    local_values = sorted_distances[:, :usable]
    local_scale = np.mean(local_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized = safe / np.maximum(pair_scale, np.finfo(float).eps)
    distance_score = np.exp(-np.clip(normalized ** 1.36, 0.0, 50.0))

    row_affinity = np.exp(
        -np.clip((safe / local_scale[:, None]) ** 1.22, 0.0, 50.0)
    )
    col_affinity = np.exp(
        -np.clip((safe / local_scale[None, :]) ** 1.22, 0.0, 50.0)
    )
    endpoint_affinity = np.sqrt(row_affinity * col_affinity)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    nearest_row = np.argmin(safe, axis=1)
    nearest_col = np.argmin(safe, axis=0)

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    mutual_top_neighbors = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    isolation_ratio = pair_scale / (safe + global_scale)
    isolation_bonus = 1.0 + 0.035 * np.clip(isolation_ratio, 0.0, 1.0)

    scores = (
        distance_score
        * endpoint_affinity
        * (0.52 + 0.48 * reciprocal_rank)
        * (
            1.0
            + 0.78 * mutual_nearest
            + 0.32 * mutual_top_neighbors
        )
        * isolation_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
