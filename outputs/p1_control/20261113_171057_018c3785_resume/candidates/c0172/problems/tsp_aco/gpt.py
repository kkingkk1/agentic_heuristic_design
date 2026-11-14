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

    if not valid.any():
        return result

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        distance_scale = float(np.median(positive_values))
    else:
        distance_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    distance_scale = max(distance_scale, eps)

    masked = np.where(valid, d, np.inf)

    row_sorted = np.partition(masked, 1, axis=1)
    col_sorted = np.partition(masked, 1, axis=0)

    row_best = row_sorted[:, 0]
    row_second = row_sorted[:, 1]
    col_best = col_sorted[0, :]
    col_second = col_sorted[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_positive_gap = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive_gap = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive_gap))
        if row_positive_gap.size
        else distance_scale
    )
    col_gap_scale = (
        float(np.median(col_positive_gap))
        if col_positive_gap.size
        else distance_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    with np.errstate(over="ignore", invalid="ignore"):
        row_regret = np.exp(
            -np.maximum(row_excess, 0.0) / row_gap_scale
        )
        col_regret = np.exp(
            -np.maximum(col_excess, 0.0) / col_gap_scale
        )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.35 * distance_scale, eps)

    affinity = np.zeros_like(d)
    np.divide(
        row_excess,
        temperature,
        out=affinity,
        where=valid,
    )

    np.minimum(affinity, 60.0, out=affinity)
    np.negative(affinity, out=affinity)

    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(affinity, out=affinity)

    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    for _ in range(32):
        row_mass = affinity @ col_scale
        np.divide(
            1.0,
            row_mass,
            out=row_scale,
            where=row_mass > 0.0,
        )
        row_scale[row_mass <= 0.0] = 0.0

        col_mass = affinity.T @ row_scale
        np.divide(
            1.0,
            col_mass,
            out=col_scale,
            where=col_mass > 0.0,
        )
        col_scale[col_mass <= 0.0] = 0.0

    np.multiply(affinity, row_scale[:, None], out=result)
    result *= col_scale[None, :]

    with np.errstate(over="ignore", invalid="ignore"):
        result *= np.power(regret + eps, 0.7)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(result.max())
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result
