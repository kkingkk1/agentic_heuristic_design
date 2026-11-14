import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-priority scores for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    valid_directed = finite & nonnegative
    valid = valid_directed & valid_directed.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clean = np.where(finite, distances, 0.0)
    symmetric = 0.5 * (clean + clean.T)
    safe = np.where(valid, symmetric, np.inf)

    positive = symmetric[valid & (symmetric > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = float(np.median(symmetric[valid]))

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    neighbor_count = min(3, n - 1)
    nearest = np.partition(
        safe,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)
    sums = np.where(finite_nearest, nearest, 0.0).sum(axis=1)

    local_scale = np.divide(
        sums,
        np.maximum(counts, 1),
        dtype=float,
    )
    local_scale = np.where(counts > 0, local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = np.divide(
        safe,
        endpoint_scale,
        out=np.full_like(safe, np.inf, dtype=float),
        where=endpoint_scale > 0.0,
    )
    normalized_distance = np.minimum(normalized_distance, 60.0)
    local_affinity = np.exp(-normalized_distance)

    global_distance = np.divide(
        safe,
        global_scale,
        out=np.full_like(safe, np.inf, dtype=float),
    )
    global_distance = np.minimum(global_distance, 60.0)
    global_affinity = 1.0 / (1.0 + global_distance * global_distance)

    row_relative = np.divide(
        safe,
        local_scale[:, None],
        out=np.full_like(safe, np.inf, dtype=float),
    )
    col_relative = np.divide(
        safe,
        local_scale[None, :],
        out=np.full_like(safe, np.inf, dtype=float),
    )

    scarcity = 1.0 / np.sqrt(
        (1.0 + row_relative) * (1.0 + col_relative)
    )

    nearest_row = np.argmin(safe, axis=1)
    nearest_col = np.argmin(safe, axis=0)
    indices = np.arange(n)

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == indices[None, :])
        & (nearest_col[None, :] == indices[:, None])
    )

    top_k = min(2, n - 1)
    row_candidates = np.argpartition(
        safe,
        kth=top_k - 1,
        axis=1,
    )[:, :top_k]
    col_candidates = np.argpartition(
        safe,
        kth=top_k - 1,
        axis=0,
    )[:top_k, :]

    row_top = np.zeros((n, n), dtype=bool)
    col_top = np.zeros((n, n), dtype=bool)

    row_top[indices[:, None], row_candidates] = True
    col_top[col_candidates, indices[None, :]] = True

    mutual_top_two = valid & row_top & col_top

    isolation_ratio = np.divide(
        endpoint_scale,
        safe + global_scale,
        out=np.zeros_like(safe, dtype=float),
        where=np.isfinite(safe),
    )
    isolation_ratio = np.clip(isolation_ratio, 0.0, 1.0)

    score = (
        local_affinity
        * (0.55 + 0.45 * scarcity)
        * (0.75 + 0.25 * global_affinity)
        * (1.0 + 0.30 * mutual_nearest.astype(float))
        * (1.0 + 0.10 * mutual_top_two.astype(float))
        * (1.0 + 0.20 * isolation_ratio)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
