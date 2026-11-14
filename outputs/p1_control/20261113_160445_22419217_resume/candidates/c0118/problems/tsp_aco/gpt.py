import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
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

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
    scale = max(scale, np.finfo(float).eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    finite_row_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    finite_col_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(finite_row_gaps))
        if finite_row_gaps.size
        else scale
    )
    col_gap_scale = (
        float(np.median(finite_col_gaps))
        if finite_col_gaps.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, np.finfo(float).eps)
    col_gap_scale = max(col_gap_scale, np.finfo(float).eps)

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

    row_regret = np.exp(
        -np.maximum(row_excess, 0.0) / row_gap_scale
    )
    col_regret = np.exp(
        -np.maximum(col_excess, 0.0) / col_gap_scale
    )
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.35 * scale, np.finfo(float).eps)
    relative_distance = np.zeros_like(d)
    np.divide(
        row_excess,
        temperature,
        out=relative_distance,
        where=valid,
    )

    affinity = np.exp(-np.minimum(relative_distance, 60.0))
    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(32):
            row_mass = affinity @ col_scale
            np.divide(
                1.0,
                row_mass,
                out=row_scale,
                where=row_mass > 0.0,
            )
            row_scale[row_mass <= 0.0] = 0.0

            col_mass = affinity.T @ row_scale
            np.divide(
                1.0,
                col_mass,
                out=col_scale,
                where=col_mass > 0.0,
            )
            col_scale[col_mass <= 0.0] = 0.0

    balanced = affinity * row_scale[:, None] * col_scale[None, :]
    score = balanced * np.power(
        regret + np.finfo(float).eps,
        0.7,
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
