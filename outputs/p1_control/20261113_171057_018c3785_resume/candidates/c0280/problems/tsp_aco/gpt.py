import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return regret-aware, globally balanced edge priors for directed TSP."""
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

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    row_has = valid.any(axis=1)
    col_has = valid.any(axis=0)

    row_best = np.full(n, scale, dtype=float)
    col_best = np.full(n, scale, dtype=float)
    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    row_best[row_has] = np.min(masked[row_has], axis=1)
    col_best[col_has] = np.min(masked[:, col_has], axis=0)

    row_sorted = np.sort(masked, axis=1)
    col_sorted = np.sort(masked, axis=0)

    row_second[row_has] = row_sorted[row_has, 1]
    col_second[col_has] = col_sorted[1, col_has]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_gap[~np.isfinite(row_gap)] = 0.0
    col_gap[~np.isfinite(col_gap)] = 0.0

    row_gap_values = row_gap[row_gap > 0.0]
    col_gap_values = col_gap[col_gap > 0.0]

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

    if not np.isfinite(row_gap_scale) or row_gap_scale <= 0.0:
        row_gap_scale = scale
    if not np.isfinite(col_gap_scale) or col_gap_scale <= 0.0:
        col_gap_scale = scale

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

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_regret = np.exp(
            -np.clip(row_excess / row_gap_scale, 0.0, 60.0)
        )
        col_regret = np.exp(
            -np.clip(col_excess / col_gap_scale, 0.0, 60.0)
        )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0
    regret[~np.isfinite(regret)] = 0.0

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
    affinity[~np.isfinite(affinity)] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(48):
            row_mass = affinity @ col_scale
            row_scale.fill(0.0)
            np.divide(
                1.0,
                row_mass,
                out=row_scale,
                where=np.isfinite(row_mass) & (row_mass > 0.0),
            )

            col_mass = affinity.T @ row_scale
            col_scale.fill(0.0)
            np.divide(
                1.0,
                col_mass,
                out=col_scale,
                where=np.isfinite(col_mass) & (col_mass > 0.0),
            )

    np.multiply(affinity, row_scale[:, None], out=result)
    result *= col_scale[None, :]

    with np.errstate(over="ignore", invalid="ignore"):
        result *= np.power(regret + eps, 0.8)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
