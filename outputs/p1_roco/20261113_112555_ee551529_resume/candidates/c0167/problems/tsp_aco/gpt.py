import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-priority scores for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances) & (distances >= 0.0)
    valid = finite & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    # Build a robust symmetric view while preserving available directed values.
    transpose = distances.T
    both_finite = np.isfinite(distances) & np.isfinite(transpose)
    symmetric = np.where(
        both_finite,
        0.5 * (distances + transpose),
        np.where(np.isfinite(distances), distances, transpose),
    )
    symmetric_valid = np.isfinite(symmetric) & (symmetric >= 0.0)
    symmetric_valid &= ~np.eye(n, dtype=bool)

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    safe = np.where(symmetric_valid, symmetric, np.inf)

    positive = symmetric[symmetric_valid & (symmetric > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = float(np.median(symmetric[symmetric_valid]))

    global_scale = max(global_scale, eps)

    # Local scale from the nearest finite neighbors.
    neighbor_count = min(4, n - 1)
    nearest = np.partition(safe, neighbor_count - 1, axis=1)[:, :neighbor_count]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized = safe / np.maximum(endpoint_scale, eps)
    normalized = np.clip(normalized, 0.0, 60.0)

    # Smooth local and global distance desirability.
    local_desirability = np.exp(-normalized)
    global_desirability = 1.0 / (
        1.0 + np.clip(safe / global_scale, 0.0, 60.0) ** 2
    )

    # Stable ranks among valid neighbors only.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    row_positions = np.arange(n, dtype=float)
    col_positions = np.arange(n, dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = row_positions[None, :]
    col_rank[col_order, np.arange(n)[None, :]] = col_positions[:, None]

    rank_signal = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    rank_signal = np.where(symmetric_valid, rank_signal, 0.0)

    # Reinforce edges that are important in both endpoint neighborhoods.
    top_k = min(3, n - 1)
    row_top = np.zeros((n, n), dtype=bool)
    col_top = np.zeros((n, n), dtype=bool)

    row_top[np.arange(n)[:, None], row_order[:, :top_k]] = True
    col_top[col_order[:top_k, :], np.arange(n)[None, :]] = True

    mutual_top = row_top & col_top & symmetric_valid
    mutual_nearest = (
        mutual_top
        & (row_rank == 0.0)
        & (col_rank == 0.0)
    )

    # Mildly protect endpoints whose alternatives are comparatively expensive.
    endpoint_pressure = endpoint_scale / (
        endpoint_scale + global_scale
    )
    endpoint_pressure = np.clip(endpoint_pressure, 0.0, 1.0)

    score = (
        local_desirability
        * (0.50 + 0.50 * rank_signal)
        * (0.72 + 0.28 * global_desirability)
        * (1.0 + 0.24 * mutual_top.astype(float))
        * (1.0 + 0.20 * mutual_nearest.astype(float))
        * (1.0 + 0.18 * endpoint_pressure)
    )

    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
