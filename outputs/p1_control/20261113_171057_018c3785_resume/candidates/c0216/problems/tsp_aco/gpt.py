import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware edge priors for directed TSP."""
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
    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

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

    row_excess = np.maximum(d - row_best[:, None], 0.0)
    col_excess = np.maximum(d - col_best[None, :], 0.0)

    row_regret = np.exp(-row_excess / row_gap_scale)
    col_regret = np.exp(-col_excess / col_gap_scale)
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.32 * scale, eps)
    relative_cost = row_excess / temperature

    affinity = np.exp(-np.minimum(relative_cost, 60.0))
    affinity[~valid] = 0.0

    order = np.argsort(masked, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.25, 0.8 * np.sqrt(float(n)))
    rank_term = np.exp(
        -0.45 * (ranks / rank_scale + ranks.T / rank_scale)
    )
    rank_term[~valid] = 0.0

    nearest = np.argmin(masked, axis=1)
    mutual = nearest[nearest] == np.arange(n)

    mutual_bonus = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    mutual_bonus[rows, nearest] = mutual.astype(float)

    valid_reverse = valid[nearest, rows]
    mutual_bonus[nearest[valid_reverse], rows[valid_reverse]] = (
        mutual[valid_reverse].astype(float)
    )

    score = affinity
    score *= np.power(regret + eps, 0.75)
    score *= rank_term
    score *= 1.0 + 0.25 * mutual_bonus
    score[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(48):
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

    balanced = score * row_scale[:, None] * col_scale[None, :]
    balanced[~np.isfinite(balanced)] = 0.0
    balanced[~valid] = 0.0

    maximum = float(np.max(balanced))
    if maximum > 0.0:
        balanced /= maximum

    np.fill_diagonal(balanced, 0.0)
    return balanced.astype(float, copy=False)
