import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware priors for directed TSP edges."""
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
    positive = values[values > 0.0]

    scale = (
        float(np.median(positive))
        if positive.size
        else max(float(np.median(values)), 1.0)
    )
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_two = np.partition(masked, 1, axis=1)
    col_two = np.partition(masked, 1, axis=0)

    row_second = row_two[:, 1]
    col_second = col_two[1, :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

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

    row_excess = np.maximum(row_excess, 0.0)
    col_excess = np.maximum(col_excess, 0.0)

    temperature = max(0.30 * scale, eps)

    row_affinity = np.exp(
        -np.clip(row_excess / temperature, 0.0, 60.0)
    )
    col_affinity = np.exp(
        -np.clip(col_excess / temperature, 0.0, 60.0)
    )

    affinity = np.sqrt(row_affinity * col_affinity)
    affinity[~valid] = 0.0

    row_regret = np.exp(
        -np.clip(row_excess / row_gap_scale, 0.0, 60.0)
    )
    col_regret = np.exp(
        -np.clip(col_excess / col_gap_scale, 0.0, 60.0)
    )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    for _ in range(40):
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

    np.multiply(
        affinity,
        row_scale[:, None],
        out=result,
    )
    result *= col_scale[None, :]

    result *= np.power(regret + eps, 0.8)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
