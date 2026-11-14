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

    result = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    values = d[valid]
    positive = values[values > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(masked, axis=1)
    col_min = np.min(masked, axis=0)

    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_two = np.partition(masked, kth=1, axis=1)[:, :2]
    col_two = np.partition(masked, kth=1, axis=0)[:2, :]

    row_second = row_two[:, 1]
    col_second = col_two[1, :]

    row_gap = np.where(
        row_count >= 2,
        np.maximum(row_second - row_min, 0.0),
        scale,
    )
    col_gap = np.where(
        col_count >= 2,
        np.maximum(col_second - col_min, 0.0),
        scale,
    )

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
        row_min[:, None],
        out=row_excess,
        where=valid,
    )
    np.subtract(
        d,
        col_min[None, :],
        out=col_excess,
        where=valid,
    )

    np.maximum(row_excess, 0.0, out=row_excess, where=valid)
    np.maximum(col_excess, 0.0, out=col_excess, where=valid)

    regret_exponent = -0.5 * (
        row_excess / row_gap_scale
        + col_excess / col_gap_scale
    )
    regret = np.exp(np.clip(regret_exponent, -60.0, 0.0))
    regret[~valid] = 0.0

    temperature = max(0.22 * scale, eps)
    affinity_exponent = -(
        0.7 * row_excess + 0.3 * col_excess
    ) / temperature
    affinity = np.exp(np.clip(affinity_exponent, -60.0, 0.0))
    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for iteration in range(40):
            row_mass = affinity @ col_scale
            new_row_scale = np.zeros_like(row_scale)
            np.divide(
                1.0,
                row_mass,
                out=new_row_scale,
                where=row_mass > 0.0,
            )

            col_mass = affinity.T @ new_row_scale
            new_col_scale = np.zeros_like(col_scale)
            np.divide(
                1.0,
                col_mass,
                out=new_col_scale,
                where=col_mass > 0.0,
            )

            row_scale = new_row_scale
            col_scale = new_col_scale

            if iteration >= 5:
                balanced = (
                    affinity
                    * row_scale[:, None]
                    * col_scale[None, :]
                )
                row_error = np.max(
                    np.abs(np.sum(balanced, axis=1) - 1.0)
                )
                col_error = np.max(
                    np.abs(np.sum(balanced, axis=0) - 1.0)
                )
                if max(row_error, col_error) <= 1e-4:
                    break

    balanced = affinity * row_scale[:, None] * col_scale[None, :]

    score = balanced * np.power(regret + eps, 0.9)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
