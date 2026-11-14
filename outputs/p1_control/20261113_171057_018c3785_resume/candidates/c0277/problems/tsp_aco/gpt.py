import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    tiny = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, tiny)

    masked = np.where(valid, d, np.inf)
    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_best = np.full(n, scale, dtype=float)
    col_best = np.full(n, scale, dtype=float)
    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    rows_with_edges = row_count > 0
    cols_with_edges = col_count > 0
    rows_with_two = row_count >= 2
    cols_with_two = col_count >= 2

    row_best[rows_with_edges] = np.min(
        masked[rows_with_edges],
        axis=1,
    )
    col_best[cols_with_edges] = np.min(
        masked[:, cols_with_edges],
        axis=0,
    )

    if np.any(rows_with_two):
        row_second[rows_with_two] = np.partition(
            masked[rows_with_two],
            kth=1,
            axis=1,
        )[:, 1]

    if np.any(cols_with_two):
        col_second[cols_with_two] = np.partition(
            masked[:, cols_with_two],
            kth=1,
            axis=0,
        )[1, :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)
    row_gap[~np.isfinite(row_gap)] = 0.0
    col_gap[~np.isfinite(col_gap)] = 0.0

    row_positive_gaps = row_gap[row_gap > 0.0]
    col_positive_gaps = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(row_positive_gaps))
        if row_positive_gaps.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_positive_gaps))
        if col_positive_gaps.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(
        d,
        row_best[:, None],
        out=row_excess,
        where=valid,
    )
    np.subtract(
        d,
        col_best[None, :],
        out=col_excess,
        where=valid,
    )

    row_excess = np.maximum(row_excess, 0.0)
    col_excess = np.maximum(col_excess, 0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        row_regret = np.exp(
            -np.clip(row_excess / row_gap_scale, 0.0, 60.0)
        )
        col_regret = np.exp(
            -np.clip(col_excess / col_gap_scale, 0.0, 60.0)
        )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.24 * scale, tiny)
    relative_distance = np.zeros_like(d)

    np.divide(
        row_excess,
        temperature,
        out=relative_distance,
        where=valid,
    )

    affinity = np.exp(-np.clip(relative_distance, 0.0, 60.0))
    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    balanced = affinity.copy()

    for _ in range(32):
        previous = balanced.copy()

        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(row_sum) & (row_sum > 0.0),
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(col_sum) & (col_sum > 0.0),
        )

        balanced[~np.isfinite(balanced)] = 0.0

        if np.max(np.abs(balanced - previous)) < 1e-10:
            break

    combined = 0.70 * balanced + 0.30 * affinity

    with np.errstate(over="ignore", invalid="ignore"):
        score = combined * np.power(regret + tiny, 0.85)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
