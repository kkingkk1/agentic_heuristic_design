import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return fast, locally balanced priors for directed TSP edges."""
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

    row_second = np.partition(masked, min(1, n - 1), axis=1)[:, min(1, n - 1)]
    col_second = np.partition(masked, min(1, n - 1), axis=0)[min(1, n - 1), :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

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

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    temperature = max(0.28 * scale, eps)

    row_affinity = np.exp(-np.minimum(row_excess / temperature, 60.0))
    col_affinity = np.exp(-np.minimum(col_excess / temperature, 60.0))

    local_affinity = np.sqrt(row_affinity * col_affinity)

    row_regret = np.exp(
        -np.minimum(np.maximum(row_excess, 0.0) / row_gap_scale, 60.0)
    )
    col_regret = np.exp(
        -np.minimum(np.maximum(col_excess, 0.0) / col_gap_scale, 60.0)
    )
    regret = np.sqrt(row_regret * col_regret)

    score = local_affinity * np.power(regret + eps, 0.75)
    score[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)
    previous_row_scale = row_scale.copy()
    previous_col_scale = col_scale.copy()

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(12):
            row_mass = score @ col_scale
            row_scale.fill(0.0)
            np.divide(
                1.0,
                row_mass,
                out=row_scale,
                where=row_mass > 0.0,
            )

            col_mass = score.T @ row_scale
            col_scale.fill(0.0)
            np.divide(
                1.0,
                col_mass,
                out=col_scale,
                where=col_mass > 0.0,
            )

            row_change = np.max(
                np.abs(row_scale - previous_row_scale)
            )
            col_change = np.max(
                np.abs(col_scale - previous_col_scale)
            )

            previous_row_scale[:] = row_scale
            previous_col_scale[:] = col_scale

            if max(row_change, col_change) < 1e-3:
                break

    balanced = score * row_scale[:, None] * col_scale[None, :]
    balanced[~valid] = 0.0
    balanced[~np.isfinite(balanced)] = 0.0

    maximum = float(np.max(balanced))
    if maximum > 0.0:
        balanced /= maximum

    np.fill_diagonal(balanced, 0.0)
    return balanced.astype(float, copy=False)
