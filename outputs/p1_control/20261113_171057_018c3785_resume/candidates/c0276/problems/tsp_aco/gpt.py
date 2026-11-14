import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, endpoint-aware priors for directed TSP edges."""
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
    values = d[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(values)), 1.0)
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_positive = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive)) if row_positive.size else scale
    )
    col_gap_scale = (
        float(np.median(col_positive)) if col_positive.size else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    endpoint_excess = 0.5 * (row_excess + col_excess)
    temperature = max(0.30 * scale, eps)

    affinity = np.zeros_like(d)
    np.divide(
        endpoint_excess,
        temperature,
        out=affinity,
        where=valid,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        np.minimum(affinity, 60.0, out=affinity)
        np.negative(affinity, out=affinity)
        np.exp(affinity, out=affinity)

    affinity[~valid] = 0.0

    row_regret = np.zeros_like(d)
    col_regret = np.zeros_like(d)

    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(
            row_excess,
            row_gap_scale,
            out=row_regret,
            where=valid,
        )
        np.divide(
            col_excess,
            col_gap_scale,
            out=col_regret,
            where=valid,
        )

        np.minimum(row_regret, 60.0, out=row_regret)
        np.minimum(col_regret, 60.0, out=col_regret)

        np.negative(row_regret, out=row_regret)
        np.negative(col_regret, out=col_regret)

        np.exp(row_regret, out=row_regret)
        np.exp(col_regret, out=col_regret)

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(56):
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

    balanced = affinity * row_scale[:, None]
    balanced *= col_scale[None, :]

    with np.errstate(over="ignore", invalid="ignore"):
        result = balanced * np.power(regret + eps, 0.82)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0 and np.isfinite(maximum):
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
