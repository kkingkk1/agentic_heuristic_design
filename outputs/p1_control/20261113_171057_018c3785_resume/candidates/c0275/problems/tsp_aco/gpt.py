import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, endpoint-aware priors for directed TSP edges."""
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

    row_has = np.any(valid, axis=1)
    col_has = np.any(valid, axis=0)

    row_best = np.full(n, scale, dtype=float)
    col_best = np.full(n, scale, dtype=float)
    row_best[row_has] = np.min(masked[row_has], axis=1)
    col_best[col_has] = np.min(masked[:, col_has], axis=0)

    row_sorted = np.sort(masked, axis=1)
    col_sorted = np.sort(masked, axis=0)

    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    row_second[row_has] = row_sorted[row_has, 1]
    col_second[col_has] = col_sorted[1, col_has]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_positive = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive)) if row_positive.size else scale
    )
    col_gap_scale = (
        float(np.median(col_positive)) if col_positive.size else scale
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

    row_scaled = np.minimum(row_excess / row_gap_scale, 60.0)
    col_scaled = np.minimum(col_excess / col_gap_scale, 60.0)

    row_regret = np.exp(-row_scaled)
    col_regret = np.exp(-col_scaled)
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    endpoint_excess = 0.5 * (row_excess + col_excess)
    temperature = max(0.32 * scale, eps)

    affinity = np.exp(
        -np.minimum(endpoint_excess / temperature, 60.0)
    )
    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(48):
            row_mass = affinity @ col_scale
            row_scale = np.divide(
                1.0,
                row_mass,
                out=np.zeros_like(row_mass),
                where=np.isfinite(row_mass) & (row_mass > 0.0),
            )

            col_mass = affinity.T @ row_scale
            col_scale = np.divide(
                1.0,
                col_mass,
                out=np.zeros_like(col_mass),
                where=np.isfinite(col_mass) & (col_mass > 0.0),
            )

    result = affinity * row_scale[:, None] * col_scale[None, :]
    result *= np.power(regret + eps, 0.9)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
