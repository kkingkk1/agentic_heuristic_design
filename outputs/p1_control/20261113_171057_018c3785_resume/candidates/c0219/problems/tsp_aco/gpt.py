import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, locality- and regret-aware TSP edge priors."""
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

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(finite_values)), 1.0)

    tiny = np.finfo(float).eps
    scale = max(scale, tiny)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_gap_values = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gap_values = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_gap_values)) if row_gap_values.size else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values)) if col_gap_values.size else scale
    )

    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    row_excess = np.maximum(row_excess, 0.0)
    col_excess = np.maximum(col_excess, 0.0)

    locality = np.exp(
        -0.5
        * (
            row_excess / max(0.38 * scale, tiny)
            + col_excess / max(0.38 * scale, tiny)
        )
    )
    locality[~valid] = 0.0

    row_regret = np.exp(-row_excess / row_gap_scale)
    col_regret = np.exp(-col_excess / col_gap_scale)
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.34 * scale, tiny)
    affinity = np.exp(-np.minimum(row_excess / temperature, 60.0))
    affinity *= np.power(regret + tiny, 0.78)
    affinity *= np.power(locality + tiny, 0.42)
    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(64):
            row_mass = affinity @ col_scale
            row_scale = np.divide(
                1.0,
                row_mass,
                out=np.zeros_like(row_mass),
                where=row_mass > 0.0,
            )

            col_mass = affinity.T @ row_scale
            col_scale = np.divide(
                1.0,
                col_mass,
                out=np.zeros_like(col_mass),
                where=col_mass > 0.0,
            )

    score = affinity * row_scale[:, None] * col_scale[None, :]
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
