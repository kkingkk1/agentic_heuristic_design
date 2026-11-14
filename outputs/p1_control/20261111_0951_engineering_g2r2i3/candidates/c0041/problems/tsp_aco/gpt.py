import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-scaled reciprocal-distance priorities for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive_valid = valid & (d > 0.0)
    row_scale = np.min(
        np.where(positive_valid, d, np.inf),
        axis=1,
        keepdims=True,
    )
    row_scale = np.where(np.isfinite(row_scale), row_scale, 1.0)

    score = np.zeros_like(d)
    zero_valid = valid & (d == 0.0)
    score[zero_valid] = 1.0
    score[positive_valid] = row_scale.repeat(n, axis=1)[positive_valid] / d[positive_valid]

    return score
