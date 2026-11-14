import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bidirectionally balanced Gibbs priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    safe = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    distance_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    distance_scale = max(distance_scale, np.finfo(float).eps)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_excess = np.zeros_like(d, dtype=float)
    col_excess = np.zeros_like(d, dtype=float)

    finite_rows = np.isfinite(row_min)
    finite_cols = np.isfinite(col_min)

    if np.any(finite_rows):
        row_excess[finite_rows] = np.maximum(
            d[finite_rows] - row_min[finite_rows, None],
            0.0,
        )

    if np.any(finite_cols):
        col_excess[:, finite_cols] = np.maximum(
            d[:, finite_cols] - col_min[None, finite_cols],
            0.0,
        )

    temperature = max(
        0.30 * distance_scale,
        np.finfo(float).eps,
    )

    exponent = 0.5 * (
        row_excess / temperature
        + col_excess / temperature
    )

    score = np.zeros_like(d, dtype=float)
    score[valid] = np.exp(-np.clip(exponent[valid], 0.0, 60.0))
    score[~np.isfinite(score)] = 0.0
    score[~valid] = 0.0

    for _ in range(4):
        row_sum = np.sum(score, axis=1, keepdims=True)
        score = np.divide(
            score,
            np.where(row_sum > 0.0, row_sum, 1.0),
        )
        score[~valid] = 0.0

        col_sum = np.sum(score, axis=0, keepdims=True)
        score = np.divide(
            score,
            np.where(col_sum > 0.0, col_sum, 1.0),
        )
        score[~valid] = 0.0

    score[~np.isfinite(score)] = 0.0
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
