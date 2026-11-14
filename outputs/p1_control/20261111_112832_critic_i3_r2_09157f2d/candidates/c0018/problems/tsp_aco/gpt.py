import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be a square 2D array")

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    finite = np.isfinite(d)

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    finite_values = d[finite]
    max_finite = max(float(np.max(finite_values)), 0.0)
    large_value = max_finite * 10.0 + 1.0

    d[~finite] = large_value
    np.maximum(d, 0.0, out=d)
    np.fill_diagonal(d, large_value)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive_values = d[off_diagonal]
    scale = float(np.median(positive_values))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive_values))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    epsilon = max(scale * 1e-12, np.finfo(float).eps)

    score = np.reciprocal(d + epsilon)
    np.fill_diagonal(score, 0.0)

    score += score.T
    score *= 0.5

    row_max = np.max(score, axis=1, keepdims=True)
    np.divide(
        score,
        row_max,
        out=score,
        where=row_max > 0.0,
    )

    np.fill_diagonal(score, 0.0)
    return score
