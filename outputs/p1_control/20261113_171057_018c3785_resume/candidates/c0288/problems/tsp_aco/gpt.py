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

    if not valid.any():
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
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

    row_excess = np.where(valid, np.maximum(d - row_best[:, None], 0.0), 0.0)
    col_excess = np.where(valid, np.maximum(d - col_best[None, :], 0.0), 0.0)

    regret = np.exp(
        -0.5
        * (
            row_excess / row_gap_scale
            + col_excess / col_gap_scale
        )
    )
    regret[~valid] = 0.0

    temperature = max(0.24 * scale, eps)
    affinity = np.exp(-np.minimum(row_excess / temperature, 60.0))
    affinity[~valid] = 0.0

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

    score = (
        affinity
        * row_scale[:, None]
        * col_scale[None, :]
        * np.power(regret + eps, 0.75)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
