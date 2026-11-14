import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    work = d.copy()
    work[~np.isfinite(work)] = np.inf
    np.fill_diagonal(work, np.inf)
    work = np.maximum(work, 0.0)

    finite = work[np.isfinite(work)]
    if finite.size == 0:
        return np.zeros_like(work)

    scale = np.median(finite)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(finite[finite > 0]) if np.any(finite > 0) else 1.0
    scale = max(scale, np.finfo(float).eps)

    positive = np.where(work > 0, work, scale)
    inverse = 1.0 / positive
    inverse /= np.max(inverse[np.isfinite(inverse)])

    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    rank_score = 1.0 - ranks / max(n - 1, 1)

    nearest = np.argmin(work, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0

    row_mean = np.mean(np.where(np.isfinite(work), work, scale), axis=1)
    isolation = row_mean / (row_mean + scale)
    isolation_pair = 0.5 * (isolation[:, None] + isolation[None, :])

    closeness = np.exp(-positive / scale)

    score = (
        0.45 * inverse
        + 0.25 * rank_score
        + 0.15 * closeness
        + 0.10 * isolation_pair
        + 0.05 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
