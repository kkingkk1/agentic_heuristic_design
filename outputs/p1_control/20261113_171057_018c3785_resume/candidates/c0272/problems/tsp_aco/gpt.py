import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware edge priors for directed TSP edges."""
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
        scale = float(np.median(values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_gaps)) if row_gaps.size else scale
    )
    col_gap_scale = (
        float(np.median(col_gaps)) if col_gaps.size else scale
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

    row_excess[~valid] = 0.0
    col_excess[~valid] = 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        row_regret = np.exp(
            -np.clip(row_excess / row_gap_scale, 0.0, 60.0)
        )
        col_regret = np.exp(
            -np.clip(col_excess / col_gap_scale, 0.0, 60.0)
        )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.30 * scale, eps)
    local_affinity = np.zeros_like(d)

    np.divide(
        row_excess,
        temperature,
        out=local_affinity,
        where=valid,
    )
    np.minimum(local_affinity, 60.0, out=local_affinity)
    np.negative(local_affinity, out=local_affinity)

    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(local_affinity, out=local_affinity)

    local_affinity[~valid] = 0.0
    local_affinity[~np.isfinite(local_affinity)] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(32):
            row_mass = local_affinity @ col_scale
            row_scale.fill(0.0)
            np.divide(
                1.0,
                row_mass,
                out=row_scale,
                where=np.isfinite(row_mass) & (row_mass > 0.0),
            )

            col_mass = local_affinity.T @ row_scale
            col_scale.fill(0.0)
            np.divide(
                1.0,
                col_mass,
                out=col_scale,
                where=np.isfinite(col_mass) & (col_mass > 0.0),
            )

            row_scale[~np.isfinite(row_scale)] = 0.0
            col_scale[~np.isfinite(col_scale)] = 0.0

    balanced = np.zeros_like(d)
    np.multiply(
        local_affinity,
        row_scale[:, None],
        out=balanced,
    )
    balanced *= col_scale[None, :]
    balanced[~np.isfinite(balanced)] = 0.0

    combined = 0.60 * balanced + 0.40 * local_affinity

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result[:] = combined * np.power(regret + eps, 0.80)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
