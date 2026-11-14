import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return local-cycle-completion priors for directed TSP edges."""
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
    safe = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    row_best = np.where(np.isfinite(row_best), row_best, 0.0)
    col_best = np.where(np.isfinite(col_best), col_best, 0.0)

    continuation = row_best[None, :]
    closure = col_best[:, None]

    fragment_cost = d + continuation + closure
    fragment_cost[~valid] = np.inf

    finite_costs = fragment_cost[valid]
    positive_costs = finite_costs[finite_costs > 0.0]

    if positive_costs.size:
        cost_scale = float(np.median(positive_costs))
    elif finite_costs.size:
        cost_scale = max(float(np.max(finite_costs)), 1.0)
    else:
        cost_scale = 1.0

    if not np.isfinite(cost_scale) or cost_scale <= 0.0:
        cost_scale = 1.0

    row_order = np.argsort(np.argsort(safe, axis=1), axis=1)
    col_order = np.argsort(np.argsort(safe, axis=0), axis=0)

    row_rank = np.divide(
        row_order + 1.0,
        np.maximum(row_count[:, None], 1.0),
    )
    col_rank = np.divide(
        col_order + 1.0,
        np.maximum(col_count[None, :], 1.0),
    )

    rank_penalty = 0.5 * (row_rank + col_rank)

    local_density = np.sqrt(
        np.maximum(row_count[:, None], 1.0)
        * np.maximum(col_count[None, :], 1.0)
    )
    density_factor = 1.0 / np.maximum(local_density, 1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_cost = fragment_cost / cost_scale
        base_score = 1.0 / (1.0 + np.maximum(relative_cost, 0.0))
        rank_score = 1.0 / (1.0 + rank_penalty)
        score = base_score * (0.65 + 0.35 * rank_score)
        score *= 0.9 + 0.1 * density_factor

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
