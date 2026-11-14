import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    fallback = float(np.max(finite_values)) if finite_values.size else 1.0
    d[~np.isfinite(d)] = fallback

    k = min(max(2, int(np.sqrt(n))), n - 1)
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    global_scale = np.median(d[np.triu_indices(n, 1)])

    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]

    relative_distance = d / np.sqrt(row_scale * col_scale)
    short_edge_score = 1.0 / (1.0 + relative_distance)

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks = np.maximum(ranks, 1.0)

    rank_score = 1.0 / np.sqrt(ranks)
    mutual_rank_score = np.sqrt(rank_score * rank_score.T)

    row_affinity = np.exp(-d / row_scale)
    col_affinity = np.exp(-d / col_scale)
    mutual_affinity = np.sqrt(row_affinity * col_affinity)

    score = (
        0.45 * short_edge_score
        + 0.30 * mutual_rank_score
        + 0.25 * mutual_affinity
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    finite_score = score[np.isfinite(score)]
    if finite_score.size:
        maximum = np.max(finite_score)
        if maximum > 0:
            score /= maximum

    return score
