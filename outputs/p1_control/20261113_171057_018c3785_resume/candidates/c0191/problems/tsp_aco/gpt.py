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
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_excess = np.zeros_like(d, dtype=float)
    col_excess = np.zeros_like(d, dtype=float)

    finite_rows = np.isfinite(row_min)
    finite_cols = np.isfinite(col_min)

    row_excess[finite_rows] = np.maximum(
        d[finite_rows] - row_min[finite_rows, None],
        0.0,
    )
    col_excess[:, finite_cols] = np.maximum(
        d[:, finite_cols] - col_min[None, finite_cols],
        0.0,
    )

    row_values = row_excess[valid]
    col_values = col_excess[valid]

    row_scale_values = row_values[np.isfinite(row_values) & (row_values > 0.0)]
    col_scale_values = col_values[np.isfinite(col_values) & (col_values > 0.0)]

    row_scale = (
        float(np.median(row_scale_values))
        if row_scale_values.size
        else global_scale
    )
    col_scale = (
        float(np.median(col_scale_values))
        if col_scale_values.size
        else global_scale
    )

    row_scale = max(row_scale, eps)
    col_scale = max(col_scale, eps)

    temperature_row = max(0.70 * row_scale, eps)
    temperature_col = max(0.70 * col_scale, eps)

    score = np.zeros_like(d, dtype=float)
    exponent = (
        0.5 * row_excess / temperature_row
        + 0.5 * col_excess / temperature_col
    )

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

    score = np.power(np.maximum(score, 0.0), 1.10)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
