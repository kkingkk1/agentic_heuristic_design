import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return dense, bidirectionally balanced Gibbs priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    safe_d = np.where(valid, d, np.inf)
    row_min = np.min(safe_d, axis=1)
    col_min = np.min(safe_d, axis=0)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    row_excess = np.zeros_like(d, dtype=float)
    col_excess = np.zeros_like(d, dtype=float)

    np.subtract(d, row_min[:, None], out=row_excess, where=valid)
    np.subtract(d, col_min[None, :], out=col_excess, where=valid)

    np.maximum(row_excess, 0.0, out=row_excess)
    np.maximum(col_excess, 0.0, out=col_excess)

    row_positive = row_excess[valid & (row_excess > 0.0)]
    col_positive = col_excess[valid & (col_excess > 0.0)]

    row_scale = (
        float(np.median(row_positive))
        if row_positive.size
        else global_scale
    )
    col_scale = (
        float(np.median(col_positive))
        if col_positive.size
        else global_scale
    )

    eps = np.finfo(float).eps
    temperature_row = max(0.70 * row_scale, eps)
    temperature_col = max(0.70 * col_scale, eps)

    exponent = (
        -0.5 * row_excess / temperature_row
        -0.5 * col_excess / temperature_col
    )

    np.exp(np.clip(exponent, -60.0, 0.0), out=score, where=valid)
    score[~valid] = 0.0

    for _ in range(4):
        row_sum = score.sum(axis=1)
        row_scale_factor = np.divide(
            1.0,
            row_sum,
            out=np.zeros_like(row_sum),
            where=row_sum > 0.0,
        )
        score *= row_scale_factor[:, None]
        score[~valid] = 0.0

        col_sum = score.sum(axis=0)
        col_scale_factor = np.divide(
            1.0,
            col_sum,
            out=np.zeros_like(col_sum),
            where=col_sum > 0.0,
        )
        score *= col_scale_factor[None, :]
        score[~valid] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
