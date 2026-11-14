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
        return np.zeros_like(d, dtype=float)

    scale = np.median(d[valid])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_d = np.where(valid, d, np.inf)

    normalized_d = np.zeros((n, n), dtype=float)
    normalized_d[valid] = d[valid] / scale

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / (normalized_d[valid] + 1e-12)
    inverse_distance /= np.max(inverse_distance) + 1e-12

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        rank = 0
        for j in np.argsort(safe_d[i]):
            if valid[i, j]:
                ranks[i, j] = 1.0 / (rank + 1.0)
                rank += 1

    row_max = np.max(ranks, axis=1, keepdims=True)
    row_rank = ranks / (row_max + 1e-12)
    reciprocal_rank = np.sqrt(row_rank * row_rank.T)

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if j < n and valid[i, j] and nearest[j] == i:
            mutual[i, j] = 1.0

    score = (
        0.60 * inverse_distance
        + 0.25 * reciprocal_rank
        + 0.15 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score[~valid & ~valid.T] = 0.0
    np.fill_diagonal(score, 0.0)

    positive_edges = score > 0
    score[positive_edges] += 1e-12

    return score
