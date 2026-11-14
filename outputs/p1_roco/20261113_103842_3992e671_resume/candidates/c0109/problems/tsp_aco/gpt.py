import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe_d = np.where(valid, d, np.inf)

    positive_distances = d[valid]
    global_scale = np.median(positive_distances)
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    # Robust global distance desirability.
    normalized = np.clip(d / global_scale, 0.0, 1e12)
    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / (1.0 + normalized[valid])

    # Row-wise ranks and local distance scales.
    rank_score = np.zeros((n, n), dtype=float)
    local_score = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[np.argsort(d[i, indices], kind="mergesort")]
        nearest[i] = order[0]

        ranks = np.arange(1, order.size + 1, dtype=float)
        rank_score[i, order] = np.exp(-0.22 * (ranks - 1.0))

        local_scale = np.median(d[i, order[: min(5, order.size)]])
        if not np.isfinite(local_scale) or local_scale <= 0:
            local_scale = global_scale

        local_score[i, indices] = np.exp(
            -np.minimum(d[i, indices] / (local_scale + 1e-12), 50.0)
        )

    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    # Symmetrize directed evidence before combining it.
    symmetric_valid = valid | valid.T
    inverse_distance = np.maximum(inverse_distance, inverse_distance.T)
    rank_score = np.maximum(rank_score, rank_score.T)
    local_score = np.maximum(local_score, local_score.T)
    mutual = np.maximum(mutual, mutual.T)

    score = (
        0.42 * inverse_distance
        + 0.28 * rank_score
        + 0.22 * local_score
        + 0.08 * mutual
    )

    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    score[symmetric_valid & (score <= 0)] = 1e-12
    np.fill_diagonal(score, 0.0)

    return score
