import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(d, 0.0)
    finite = np.isfinite(d)
    symmetric_finite = finite | finite.T

    both_finite = finite & finite.T
    sym_d = np.full((n, n), np.inf, dtype=float)
    sym_d[both_finite] = 0.5 * (d[both_finite] + d.T[both_finite])

    only_d = finite & ~finite.T
    sym_d[only_d] = d[only_d]
    sym_d[~finite & finite.T] = d.T[~finite & finite.T]
    np.fill_diagonal(sym_d, np.inf)

    valid = np.isfinite(sym_d)
    positive = sym_d[valid & (sym_d > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = max(float(np.median(positive)), np.finfo(float).eps)
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe = np.where(valid, sym_d, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + 0.05 * scale)
    inverse /= max(float(inverse.max()), eps)

    rank_score = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[np.argsort(safe[i, candidates], kind="stable")]
        nearest[i] = int(order[0])

        for rank, j in enumerate(order, start=1):
            rank_score[i, j] = 1.0 / rank

    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    local_scale = np.full(n, scale, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            local_scale[i] = max(float(np.median(safe[i, candidates])), eps)

    contrast = np.zeros((n, n), dtype=float)
    denominator = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    contrast[valid] = np.exp(-safe[valid] / denominator[valid])

    score = (
        0.55 * inverse
        + 0.25 * np.sqrt(rank_score * rank_score.T)
        + 0.12 * contrast
        + 0.08 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
