import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-scaled distance priorities for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, eps)

    row_scale = np.min(
        np.where(valid & (d > 0.0), d, np.inf),
        axis=1,
        keepdims=True,
    )
    row_scale = np.where(np.isfinite(row_scale), row_scale, fallback)
    row_scale = np.maximum(row_scale, eps)

    safe_distance = np.maximum(d, eps)
    score = 1.0 / (1.0 + safe_distance / row_scale)
    score *= score
    score[~valid] = 0.0

    row_max = np.max(score, axis=1, keepdims=True)
    return np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )
