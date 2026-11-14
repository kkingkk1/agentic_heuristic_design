import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(finite_positive)) if finite_positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_d = np.where(np.isfinite(d), np.maximum(d, 0.0), 10.0 * scale)

    kernel = np.exp(-safe_d / scale)
    inverse = 1.0 / (safe_d + 0.05 * scale)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1).astype(float)
    ranks /= max(n - 2, 1)
    rank_score = np.exp(-1.5 * ranks)

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i and i != j:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    row_min = np.min(safe_d, axis=1)
    col_min = np.min(safe_d, axis=0)
    savings = np.maximum(row_min[:, None] + col_min[None, :] - safe_d, 0.0)
    savings_scale = np.median(savings[savings > 0]) if np.any(savings > 0) else 1.0
    savings_score = np.tanh(savings / max(savings_scale, np.finfo(float).eps))

    score = (
        0.35 * kernel
        + 0.25 * inverse / np.max(inverse)
        + 0.20 * rank_score
        + 0.10 * savings_score
        + 0.10 * mutual
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
