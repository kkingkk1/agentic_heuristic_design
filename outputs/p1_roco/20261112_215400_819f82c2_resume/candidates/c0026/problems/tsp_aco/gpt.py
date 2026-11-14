import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)

    d = np.maximum(d, 0.0)
    finite_offdiag = np.isfinite(d) & ~np.eye(n, dtype=bool)

    if not np.any(finite_offdiag):
        return np.zeros_like(d, dtype=float)

    finite_values = d[finite_offdiag]
    fallback = float(np.max(finite_values))
    if fallback <= 0.0:
        fallback = 1.0

    safe_d = np.where(np.isfinite(d), d, fallback * 10.0)
    np.fill_diagonal(safe_d, np.inf)

    positive = safe_d[np.isfinite(safe_d) & (safe_d > 0.0)]
    distance_scale = float(np.median(positive)) if positive.size else fallback
    distance_scale = max(distance_scale, np.finfo(float).eps)

    inverse = 1.0 / (safe_d + 0.05 * distance_scale)

    row_min = np.min(safe_d, axis=1, keepdims=True)
    row_min = np.where(np.isfinite(row_min), row_min, distance_scale)
    relative_closeness = np.exp(
        -np.maximum(safe_d - row_min, 0.0) / distance_scale
    )

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i:
            mutual[i, j] = 1.0

    score = (
        0.45 * inverse / np.max(inverse[np.isfinite(inverse)])
        + 0.40 * relative_closeness
        + 0.15 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
