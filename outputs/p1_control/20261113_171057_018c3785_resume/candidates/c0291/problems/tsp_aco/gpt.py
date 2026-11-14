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
    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = max(float(np.median(positive_values)), eps)
    else:
        scale = max(float(np.median(valid_values)), 1.0)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    positive_row_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    positive_col_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(positive_row_gaps))
        if positive_row_gaps.size
        else scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gaps))
        if positive_col_gaps.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    temperature = max(0.30 * scale, eps)

    affinity = np.zeros_like(d)
    np.add(row_excess, col_excess, out=affinity)
    affinity *= -0.5 / temperature
    np.clip(affinity, -60.0, 0.0, out=affinity)

    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(affinity, out=affinity)

    affinity[~valid] = 0.0

    regret = np.zeros_like(d)
    np.add(
        row_excess / row_gap_scale,
        col_excess / col_gap_scale,
        out=regret,
        where=valid,
    )
    regret *= -0.5
    np.clip(regret, -60.0, 0.0, out=regret)

    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(regret, out=regret)

    regret[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    for _ in range(56):
        row_mass = affinity @ col_scale
        positive_rows = row_mass > 0.0
        row_scale.fill(0.0)
        np.divide(1.0, row_mass, out=row_scale, where=positive_rows)

        col_mass = affinity.T @ row_scale
        positive_cols = col_mass > 0.0
        col_scale.fill(0.0)
        np.divide(1.0, col_mass, out=col_scale, where=positive_cols)

        if positive_rows.all() and positive_cols.all():
            row_error = np.max(np.abs(row_mass - 1.0))
            col_error = np.max(np.abs(col_mass - 1.0))
            if max(row_error, col_error) < 1e-8:
                break

    result = affinity * row_scale[:, None]
    result *= col_scale[None, :]
    result *= np.power(regret + eps, 0.85)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0 and np.isfinite(maximum):
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
