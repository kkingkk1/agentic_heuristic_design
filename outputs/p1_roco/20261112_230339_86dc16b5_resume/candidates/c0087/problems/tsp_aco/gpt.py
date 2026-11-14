import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    positive_edges = finite_edges[finite_edges > 0.0]

    if positive_edges.size:
        global_scale = float(np.median(positive_edges))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    positions = np.arange(n)
    ranks[positions[:, None], order] = positions[None, :]

    local_index = min(2, n - 2)
    local_scale = d[positions, order[:, local_index]]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    distance_affinity = np.exp(-np.minimum(d, np.finfo(float).max) / pair_scale)

    rank = ranks.astype(float)
    rank_score = 1.0 / (1.0 + rank)
    rank_score = np.sqrt(rank_score * rank_score.T)

    nearest = order[:, 0]
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    score = (
        0.55 * distance_affinity
        + 0.30 * rank_score
        + 0.15 * mutual_nearest
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
