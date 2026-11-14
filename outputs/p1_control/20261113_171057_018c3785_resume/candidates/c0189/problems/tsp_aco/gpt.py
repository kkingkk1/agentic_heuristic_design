import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return fast two-sided balanced edge priors for directed TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return score

    values = d[rows, cols]

    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    tiny = np.finfo(float).eps
    temperature = max(0.30 * scale, tiny)

    row_min = np.full(n, np.inf, dtype=float)
    col_min = np.full(n, np.inf, dtype=float)
    np.minimum.at(row_min, rows, values)
    np.minimum.at(col_min, cols, values)

    row_excess = np.maximum(values - row_min[rows], 0.0)
    col_excess = np.maximum(values - col_min[cols], 0.0)

    affinity = np.exp(
        -0.5 * (row_excess + col_excess) / temperature
    )
    affinity = np.clip(affinity, tiny, 1.0)

    balanced = np.zeros((n, n), dtype=float)
    balanced[rows, cols] = affinity

    for _ in range(4):
        row_sum = np.bincount(
            rows,
            weights=affinity,
            minlength=n,
        )
        row_scale = np.divide(
            1.0,
            row_sum,
            out=np.zeros_like(row_sum),
            where=row_sum > 0.0,
        )
        affinity *= row_scale[rows]

        col_sum = np.bincount(
            cols,
            weights=affinity,
            minlength=n,
        )
        col_scale = np.divide(
            1.0,
            col_sum,
            out=np.zeros_like(col_sum),
            where=col_sum > 0.0,
        )
        affinity *= col_scale[cols]

    score[rows, cols] = affinity

    maximum = float(np.max(affinity))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    return score
