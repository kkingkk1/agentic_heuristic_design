import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced regret-aware priors for directed TSP edges."""
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
    masked = np.where(valid, d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)
    global_scale = max(global_scale, eps)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_best = np.where(row_count > 0, row_best, 0.0)
    col_best = np.where(col_count > 0, col_best, 0.0)

    row_second = np.partition(masked, kth=1, axis=1)[:, 1]
    col_second = np.partition(masked, kth=1, axis=0)[1, :]

    row_gap = np.where(
        row_count >= 2,
        np.maximum(row_second - row_best, 0.0),
        global_scale,
    )
    col_gap = np.where(
        col_count >= 2,
        np.maximum(col_second - col_best, 0.0),
        global_scale,
    )

    row_gap_values = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gap_values = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else global_scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else global_scale
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

    np.maximum(row_excess, 0.0, out=row_excess, where=valid)
    np.maximum(col_excess, 0.0, out=col_excess, where=valid)

    combined_excess = 0.5 * (row_excess + col_excess)

    temperature = max(0.24 * global_scale, eps)
    affinity = np.zeros_like(d)

    np.divide(
        combined_excess,
        temperature,
        out=affinity,
        where=valid,
    )
    np.minimum(affinity, 60.0, out=affinity)
    np.negative(affinity, out=affinity)
    np.exp(affinity, out=affinity)
    affinity[~valid] = 0.0

    regret_exponent = 0.5 * (
        row_excess / row_gap_scale
        + col_excess / col_gap_scale
    )
    regret = np.exp(-np.clip(regret_exponent, 0.0, 60.0))
    regret[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(64):
            row_mass = affinity @ col_scale
            new_row_scale = np.zeros_like(row_scale)
            np.divide(
                1.0,
                row_mass,
                out=new_row_scale,
                where=row_mass > 0.0,
            )

            col_mass = affinity.T @ new_row_scale
            new_col_scale = np.zeros_like(col_scale)
            np.divide(
                1.0,
                col_mass,
                out=new_col_scale,
                where=col_mass > 0.0,
            )

            change = max(
                float(np.max(np.abs(new_row_scale - row_scale))),
                float(np.max(np.abs(new_col_scale - col_scale))),
            )

            row_scale = new_row_scale
            col_scale = new_col_scale

            if change < 1e-10:
                break

    result = affinity * row_scale[:, None] * col_scale[None, :]
    result *= np.power(regret + eps, 0.85)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
