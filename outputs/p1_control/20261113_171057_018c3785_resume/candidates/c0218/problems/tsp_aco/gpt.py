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
    valid_distances = d[valid]
    positive = valid_distances[valid_distances > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = np.where(
        np.isfinite(row_second),
        np.maximum(row_second - row_best, 0.0),
        0.0,
    )
    col_gap = np.where(
        np.isfinite(col_second),
        np.maximum(col_second - col_best, 0.0),
        0.0,
    )

    positive_row_gap = row_gap[row_gap > 0.0]
    positive_col_gap = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(positive_row_gap))
        if positive_row_gap.size
        else scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gap))
        if positive_col_gap.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    safe_row_best = np.where(np.isfinite(row_best), row_best, 0.0)
    safe_col_best = np.where(np.isfinite(col_best), col_best, 0.0)

    row_excess = np.where(
        valid,
        np.maximum(d - safe_row_best[:, None], 0.0),
        0.0,
    )
    col_excess = np.where(
        valid,
        np.maximum(d - safe_col_best[None, :], 0.0),
        0.0,
    )

    temperature = max(0.30 * scale, eps)

    with np.errstate(over="ignore", invalid="ignore"):
        row_affinity = np.exp(
            -np.minimum(row_excess / temperature, 60.0)
        )
        col_affinity = np.exp(
            -np.minimum(col_excess / temperature, 60.0)
        )
        row_regret = np.exp(
            -np.minimum(row_excess / row_gap_scale, 60.0)
        )
        col_regret = np.exp(
            -np.minimum(col_excess / col_gap_scale, 60.0)
        )

    affinity = np.sqrt(row_affinity * col_affinity)
    regret = np.sqrt(row_regret * col_regret)

    row_unique = row_gap / (row_gap + scale)
    col_unique = col_gap / (col_gap + scale)
    uniqueness = np.sqrt(
        row_unique[:, None] * col_unique[None, :]
    )

    score = affinity * np.power(regret + eps, 0.75)
    score *= 0.75 + 0.25 * uniqueness
    score[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(40):
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

    result = score * row_scale[:, None] * col_scale[None, :]
    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result
