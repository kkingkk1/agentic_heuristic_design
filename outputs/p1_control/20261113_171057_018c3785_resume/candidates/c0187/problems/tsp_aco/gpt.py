import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware edge priors for directed TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.zeros_like(d, dtype=float)

    values = d[rows, cols]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_min = np.full(n, np.inf, dtype=float)
    col_min = np.full(n, np.inf, dtype=float)
    np.minimum.at(row_min, rows, values)
    np.minimum.at(col_min, cols, values)

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    for r, c, value in zip(rows, cols, values):
        if value < row_second[r] and value > row_min[r]:
            row_second[r] = value
        if value < col_second[c] and value > col_min[c]:
            col_second[c] = value

    row_gap = np.maximum(
        np.where(np.isfinite(row_second), row_second, row_min) - row_min,
        0.0,
    )
    col_gap = np.maximum(
        np.where(np.isfinite(col_second), col_second, col_min) - col_min,
        0.0,
    )

    positive_row_gaps = row_gap[row_gap > 0.0]
    positive_col_gaps = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(positive_row_gaps))
        if positive_row_gaps.size
        else scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gaps))
        if positive_col_gaps.size
        else scale
    )

    tiny = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    centered = values - row_min[rows]
    temperature = max(scale * 0.30, tiny)

    affinity_values = np.exp(
        -np.clip(centered / temperature, 0.0, 60.0)
    )
    regret_values = np.exp(
        -np.maximum(centered, 0.0) / row_gap_scale
    )
    regret_values *= np.exp(
        -np.maximum(values - col_min[cols], 0.0) / col_gap_scale
    )
    regret_values = np.sqrt(regret_values)

    balanced = np.zeros((n, n), dtype=float)
    balanced[rows, cols] = affinity_values

    for _ in range(12):
        row_sum = balanced.sum(axis=1)
        balanced[rows, cols] /= row_sum[rows]

        col_sum = balanced.sum(axis=0)
        balanced[rows, cols] /= col_sum[cols]

    score_values = balanced[rows, cols] * np.power(
        regret_values + tiny,
        0.8,
    )

    score = np.zeros((n, n), dtype=float)
    score[rows, cols] = score_values

    maximum = float(np.max(score_values))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    return score
