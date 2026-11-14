import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    np.fill_diagonal(positive, False)

    if not np.any(positive):
        return np.zeros_like(d, dtype=float)

    safe_d = np.where(positive, d, np.inf)

    scale = np.median(d[positive])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[positive] = 1.0 / (safe_d[positive] + 1e-12)
    inverse_distance /= np.max(inverse_distance) + 1e-12

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = safe_d[i]
        order = np.argsort(row)
        rank = 0
        for j in order:
            if positive[i, j]:
                ranks[i, j] = 1.0 / (rank + 1.0)
                rank += 1

    row_rank = ranks / (np.max(ranks, axis=1, keepdims=True) + 1e-12)
    col_rank = row_rank.T
    reciprocal_rank = np.sqrt(row_rank * col_rank)

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j < n and nearest[j] == i and positive[i, j]:
            mutual[i, j] = 1.0

    score = (
        0.55 * inverse_distance
        + 0.30 * reciprocal_rank
        + 0.15 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    if np.any(score > 0):
        score[score > 0] += 1e-12

    return score
