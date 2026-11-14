import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)

    finite_off_diag = np.isfinite(d) & ~np.eye(n, dtype=bool)
    if not np.any(finite_off_diag):
        return np.zeros_like(d, dtype=float)

    positive = d[finite_off_diag & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_d = np.where(np.isfinite(d), np.maximum(d, 0.0), np.inf)
    safe_d[np.eye(n, dtype=bool)] = np.inf

    closeness = np.exp(-np.minimum(safe_d / scale, 50.0))
    inverse = 1.0 / (safe_d + 0.05 * scale)

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i:
            mutual[i, j] = 1.0

    score = (
        0.45 * closeness
        + 0.30 * rank_score
        + 0.20 * (inverse / (np.nanmax(inverse) + np.finfo(float).eps))
        + 0.05 * mutual
    )

    score[~finite_off_diag] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
