import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = d.copy()
    np.fill_diagonal(d, np.nan)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(finite_positive)) if finite_positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    penalty = max(scale * 1e6, 1.0)
    safe_d = np.where(np.isfinite(d) & (d >= 0), d, penalty)
    safe_d = np.maximum(safe_d, 0.0)

    closeness = np.exp(-safe_d / scale)
    inverse = 1.0 / (1.0 + safe_d / scale)

    nearest = np.argmin(safe_d, axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if i != j:
            mutual_bonus[i, j] += 1.0
            if nearest[j] == i:
                mutual_bonus[i, j] += 1.0

    score = 0.55 * closeness + 0.30 * inverse + 0.15 * mutual_bonus
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    return score.astype(float, copy=False)
