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
    masked = np.where(valid, d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(finite_values))

    scale = max(scale, eps)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = float(np.median(row_gaps)) if row_gaps.size else scale
    col_gap_scale = float(np.median(col_gaps)) if col_gaps.size else scale

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        row_regret = np.exp(
            -np.maximum(row_excess, 0.0) / row_gap_scale
        )
        col_regret = np.exp(
            -np.maximum(col_excess, 0.0) / col_gap_scale
        )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.32 * scale, eps)

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

    for _ in range(48):
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

    np.multiply(affinity, row_scale[:, None], out=result)
    result *= col_scale[None, :]

    with np.errstate(over="ignore", invalid="ignore"):
        result *= np.power(regret + eps, 0.78)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result
