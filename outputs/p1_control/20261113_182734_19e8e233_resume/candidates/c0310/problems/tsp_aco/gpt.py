import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced quality priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = max(float(np.median(values)), 1.0)
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    if n > 2:
        row_second = np.partition(masked, 1, axis=1)[:, 1]
        col_second = np.partition(masked, 1, axis=0)[1, :]
    else:
        row_second = row_best.copy()
        col_second = col_best.copy()

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_positive_gap = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive_gap = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive_gap))
        if row_positive_gap.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_positive_gap))
        if col_positive_gap.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)
    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    cold_temperature = max(0.24 * scale, eps)
    warm_temperature = max(0.42 * scale, eps)

    cold = np.zeros_like(d)
    warm = np.zeros_like(d)

    np.divide(row_excess, cold_temperature, out=cold, where=valid)
    np.divide(row_excess, warm_temperature, out=warm, where=valid)

    np.minimum(cold, 55.0, out=cold)
    np.minimum(warm, 55.0, out=warm)

    np.negative(cold, out=cold)
    np.negative(warm, out=warm)
    np.exp(cold, out=cold)
    np.exp(warm, out=warm)

    row_kernel = 0.80 * cold + 0.20 * warm
    row_kernel[~valid] = 0.0

    col_kernel = np.zeros_like(d)
    np.divide(col_excess, cold_temperature, out=col_kernel, where=valid)
    np.minimum(col_kernel, 55.0, out=col_kernel)
    np.negative(col_kernel, out=col_kernel)
    np.exp(col_kernel, out=col_kernel)
    col_kernel[~valid] = 0.0

    affinity = np.sqrt(np.maximum(row_kernel * col_kernel, 0.0))

    reciprocal = valid & valid.T
    reciprocal_score = np.zeros_like(d)
    reciprocal_score[reciprocal] = np.sqrt(
        np.maximum(affinity[reciprocal] * affinity.T[reciprocal], 0.0)
    )

    affinity = np.where(
        reciprocal,
        0.82 * affinity + 0.18 * reciprocal_score,
        affinity,
    )

    row_regret = np.zeros_like(d)
    col_regret = np.zeros_like(d)

    np.divide(row_excess, row_gap_scale, out=row_regret, where=valid)
    np.divide(col_excess, col_gap_scale, out=col_regret, where=valid)

    np.minimum(row_regret, 55.0, out=row_regret)
    np.minimum(col_regret, 55.0, out=col_regret)

    np.negative(row_regret, out=row_regret)
    np.negative(col_regret, out=col_regret)
    np.exp(row_regret, out=row_regret)
    np.exp(col_regret, out=col_regret)

    regret = np.sqrt(np.maximum(row_regret * col_regret, 0.0))
    np.power(regret + eps, 0.75, out=regret)
    regret[~valid] = 0.0

    affinity *= regret
    affinity[valid] = np.maximum(affinity[valid], 1e-12)
    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(24):
            row_mass = affinity @ col_scale
            new_row_scale = np.zeros_like(row_scale)
            np.divide(
                1.0,
                row_mass,
                out=new_row_scale,
                where=row_mass > 0.0,
            )
            row_scale = np.clip(new_row_scale, 1e-12, 1e12)

            col_mass = affinity.T @ row_scale
            new_col_scale = np.zeros_like(col_scale)
            np.divide(
                1.0,
                col_mass,
                out=new_col_scale,
                where=col_mass > 0.0,
            )
            col_scale = np.clip(new_col_scale, 1e-12, 1e12)

    affinity *= row_scale[:, None]
    affinity *= col_scale[None, :]
    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    maximum = float(np.max(affinity))
    if maximum > 0.0 and np.isfinite(maximum):
        affinity /= maximum

    np.fill_diagonal(affinity, 0.0)
    return affinity.astype(float, copy=False)
