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

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(valid_values)), 1.0)
    )
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

    row_gap_scale = float(np.median(row_gaps)) if row_gaps.size else scale
    col_gap_scale = float(np.median(col_gaps)) if col_gaps.size else scale

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_delta, where=valid)
    np.subtract(d, col_best[None, :], out=col_delta, where=valid)

    np.maximum(row_delta, 0.0, out=row_delta)
    np.maximum(col_delta, 0.0, out=col_delta)

    temperature = max(0.30 * scale, eps)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        row_kernel = np.exp(-np.minimum(row_delta / temperature, 60.0))
        col_kernel = np.exp(-np.minimum(col_delta / temperature, 60.0))
        affinity = np.sqrt(row_kernel * col_kernel)

        row_regret = np.exp(-row_delta / row_gap_scale)
        col_regret = np.exp(-col_delta / col_gap_scale)
        regret = np.sqrt(row_regret * col_regret)

    affinity[~valid] = 0.0
    regret[~valid] = 0.0

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

    balanced = affinity * row_scale[:, None]
    balanced *= col_scale[None, :]

    with np.errstate(over="ignore", invalid="ignore"):
        score = balanced * (0.30 + 0.70 * np.power(regret, 0.85))

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
