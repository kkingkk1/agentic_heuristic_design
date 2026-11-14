import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    positive = d[finite & (d > 0)]
    scale = np.median(positive) if positive.size else 1.0
    scale = max(float(scale), np.finfo(float).eps)

    safe_d = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(safe_d, np.inf)

    inv = 1.0 / (safe_d + scale * 1e-6)
    inv[~finite] = 0.0

    row_min = np.min(safe_d, axis=1, keepdims=True)
    row_min[~np.isfinite(row_min)] = scale
    closeness = np.exp(-np.maximum(safe_d - row_min, 0.0) / scale)
    closeness[~finite] = 0.0

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1).astype(float)
    ranks[~finite] = float(n)
    rank_score = 1.0 / (1.0 + ranks)

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j < n and nearest[j] == i and i != j:
            mutual[i, j] = 1.0

    def normalize_rows(x: np.ndarray) -> np.ndarray:
        row_max = np.max(x, axis=1, keepdims=True)
        row_max[row_max <= 0] = 1.0
        return x / row_max

    score = (
        0.50 * normalize_rows(inv)
        + 0.25 * normalize_rows(closeness)
        + 0.15 * normalize_rows(rank_score)
        + 0.10 * mutual
    )

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
