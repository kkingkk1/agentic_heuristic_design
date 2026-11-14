import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)

    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    positive = d[valid]
    if positive.size == 0:
        return np.zeros_like(d, dtype=float)

    scale = np.median(positive)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(positive)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    normalized = np.maximum(d / scale, 0.0)

    inverse = np.zeros_like(d, dtype=float)
    inverse[valid] = 1.0 / (1.0 + normalized[valid])

    row_min = np.full(n, np.inf, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            row_min[i] = np.min(d[i, candidates])

    nearest_affinity = np.zeros_like(d, dtype=float)
    for i in range(n):
        if not np.isfinite(row_min[i]) or row_min[i] <= 0:
            continue
        candidates = np.flatnonzero(valid[i])
        nearest_affinity[i, candidates] = np.clip(
            row_min[i] / np.maximum(d[i, candidates], row_min[i]),
            0.0,
            1.0,
        )

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            nearest[i] = candidates[np.argmin(d[i, candidates])]

    mutual = np.zeros_like(d, dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    score = (
        0.55 * inverse
        + 0.30 * nearest_affinity
        + 0.15 * mutual
    )

    score = np.maximum(score, score.T)
    score[~valid & ~valid.T] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
