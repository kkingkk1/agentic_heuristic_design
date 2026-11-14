import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(valid_values))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)
    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_best = masked.min(axis=1)
    col_best = masked.min(axis=0)

    row_best = np.where(row_count > 0, row_best, 0.0)
    col_best = np.where(col_count > 0, col_best, 0.0)

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    rows_with_two = row_count >= 2
    cols_with_two = col_count >= 2

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

    row_gap = np.where(
        rows_with_two,
        np.maximum(row_second - row_best, 0.0),
        scale,
    )
    col_gap = np.where(
        cols_with_two,
        np.maximum(col_second - col_best, 0.0),
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

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

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

    np.maximum(row_excess, 0.0, out=row_excess)
    np.maximum(col_excess, 0.0, out=col_excess)

    row_regret = np.exp(
        -np.clip(row_excess / row_gap_scale, 0.0, 60.0)
    )
    col_regret = np.exp(
        -np.clip(col_excess / col_gap_scale, 0.0, 60.0)
    )
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.22 * scale, eps)
    endpoint_cost = 0.5 * (row_excess + col_excess)

    affinity = np.exp(
        -np.clip(endpoint_cost / temperature, 0.0, 60.0)
    )
    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(80):
            previous_row = row_scale.copy()
            previous_col = col_scale.copy()

            row_mass = affinity @ col_scale
            row_scale.fill(0.0)
            np.divide(
                1.0,
                row_mass,
                out=row_scale,
                where=row_mass > 0.0,
            )

            col_mass = affinity.T @ row_scale
            col_scale.fill(0.0)
            np.divide(
                1.0,
                col_mass,
                out=col_scale,
                where=col_mass > 0.0,
            )

            if (
                np.max(np.abs(row_scale - previous_row)) < 1e-11
                and np.max(np.abs(col_scale - previous_col)) < 1e-11
            ):
                break

    with np.errstate(over="ignore", invalid="ignore"):
        result = (
            affinity
            * row_scale[:, None]
            * col_scale[None, :]
            * np.power(regret + eps, 0.88)
        )

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0 and np.isfinite(maximum):
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
