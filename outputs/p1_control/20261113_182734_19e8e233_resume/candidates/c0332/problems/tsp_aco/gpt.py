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

    result = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    masked = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_best_safe = np.where(row_count > 0, row_best, 0.0)
    col_best_safe = np.where(col_count > 0, col_best, 0.0)

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    rows_two = row_count >= 2
    cols_two = col_count >= 2

    if np.any(rows_two):
        row_second[rows_two] = np.partition(
            masked[rows_two], kth=1, axis=1
        )[:, 1]

    if np.any(cols_two):
        col_second[cols_two] = np.partition(
            masked[:, cols_two], kth=1, axis=0
        )[1, :]

    row_gap = np.where(
        rows_two,
        np.maximum(row_second - row_best_safe, 0.0),
        scale,
    )
    col_gap = np.where(
        cols_two,
        np.maximum(col_second - col_best_safe, 0.0),
        scale,
    )

    row_gap_values = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gap_values = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else scale
    )

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(
        d,
        row_best_safe[:, None],
        out=row_excess,
        where=valid,
    )
    np.subtract(
        d,
        col_best_safe[None, :],
        out=col_excess,
        where=valid,
    )

    row_excess = np.maximum(row_excess, 0.0)
    col_excess = np.maximum(col_excess, 0.0)

    temperature = max(0.24 * scale, eps)

    logits = np.zeros_like(d)
    np.add(row_excess, col_excess, out=logits)
    logits *= 0.5 / temperature
    np.clip(logits, 0.0, 60.0, out=logits)

    affinity = np.exp(-logits)
    affinity[~valid] = 0.0

    row_regret = np.zeros_like(d)
    col_regret = np.zeros_like(d)

    np.divide(
        row_excess,
        max(row_gap_scale, eps),
        out=row_regret,
        where=valid,
    )
    np.divide(
        col_excess,
        max(col_gap_scale, eps),
        out=col_regret,
        where=valid,
    )

    np.clip(row_regret, 0.0, 60.0, out=row_regret)
    np.clip(col_regret, 0.0, 60.0, out=col_regret)

    regret = np.exp(-0.5 * (row_regret + col_regret))
    regret[~valid] = 0.0

    balanced = affinity.copy()

    for _ in range(64):
        previous = balanced.copy()

        row_sum = balanced.sum(axis=1, keepdims=True)
        np.divide(
            balanced,
            row_sum,
            out=balanced,
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        np.divide(
            balanced,
            col_sum,
            out=balanced,
            where=col_sum > 0.0,
        )

        balanced[~valid] = 0.0

        if np.max(np.abs(balanced - previous)) < 5e-11:
            break

    np.multiply(
        balanced,
        np.power(regret + eps, 1.12),
        out=result,
    )

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
