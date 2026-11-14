import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware and diversity-preserving TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not valid.any():
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    values = d[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = max(float(np.median(values)), 1.0)
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = masked.min(axis=1)
    col_best = masked.min(axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_positive_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

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

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    temperature = max(0.42 * scale, eps)

    affinity = np.zeros_like(d)
    np.divide(row_excess, temperature, out=affinity, where=valid)
    np.minimum(affinity, 55.0, out=affinity)
    np.negative(affinity, out=affinity)
    np.exp(affinity, out=affinity)
    affinity[~valid] = 0.0

    # Small adaptive floor: stronger only when the nearest edge dominates.
    gap_signal = np.zeros(n, dtype=float)
    finite_gap = np.isfinite(row_gap)
    gap_signal[finite_gap] = np.clip(
        row_gap[finite_gap] / row_gap_scale,
        0.0,
        1.0,
    )

    floor_strength = 0.018 + 0.032 * gap_signal
    affinity = (
        affinity * (1.0 - floor_strength[:, None])
        + valid.astype(float) * floor_strength[:, None]
    )

    regret = np.zeros_like(d)
    np.divide(row_excess, row_gap_scale, out=regret, where=valid)
    np.minimum(regret, 55.0, out=regret)
    np.negative(regret, out=regret)
    np.exp(regret, out=regret)

    col_regret = np.zeros_like(d)
    np.divide(col_excess, col_gap_scale, out=col_regret, where=valid)
    np.minimum(col_regret, 55.0, out=col_regret)
    np.negative(col_regret, out=col_regret)
    np.exp(col_regret, out=col_regret)

    regret *= col_regret
    np.sqrt(regret, out=regret)
    np.power(regret + eps, 0.8, out=regret)
    regret[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(48):
            row_mass = affinity @ col_scale
            row_scale.fill(0.0)
            np.divide(1.0, row_mass, out=row_scale, where=row_mass > 0.0)

            col_mass = affinity.T @ row_scale
            col_scale.fill(0.0)
            np.divide(1.0, col_mass, out=col_scale, where=col_mass > 0.0)

    affinity *= row_scale[:, None]
    affinity *= col_scale[None, :]
    affinity *= regret

    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    maximum = float(affinity.max())
    if maximum > 0.0:
        affinity /= maximum

    np.fill_diagonal(affinity, 0.0)
    return affinity.astype(float, copy=False)
