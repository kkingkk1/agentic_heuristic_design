import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed TSP edge priors using two-step cycle lookahead."""
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

    inf = np.inf
    masked = np.where(valid, d, inf)

    outgoing_best = np.min(masked, axis=1)
    incoming_best = np.min(masked, axis=0)

    finite_edges = d[valid]
    median_distance = float(np.median(finite_edges))
    deviations = np.abs(finite_edges - median_distance)
    mad = float(np.median(deviations))

    scale = max(1.4826 * mad, median_distance * 0.05,
                np.finfo(float).eps)

    continuation = outgoing_best[None, :]
    return_link = incoming_best[:, None]

    effective_cost = d + 0.55 * continuation + 0.35 * return_link
    effective_cost[~valid] = inf

    finite_costs = effective_cost[valid]
    cost_center = float(np.median(finite_costs))

    centered = effective_cost - cost_center
    gaussian = np.exp(-0.5 * np.square(centered / scale))
    gaussian[~valid] = 0.0

    order = np.argsort(masked, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]

    rank_affinity = np.zeros((n, n), dtype=float)
    rank_affinity[valid] = 1.0 / (
        1.0 + ranks[valid].astype(float)
    )

    local_mean = np.divide(
        np.sum(np.where(valid, d, 0.0), axis=1),
        np.maximum(np.sum(valid, axis=1), 1),
    )
    relative = np.divide(
        d,
        local_mean[:, None] + np.finfo(float).eps,
        out=np.zeros_like(d),
        where=valid,
    )
    local_affinity = np.exp(-np.minimum(relative, 50.0))
    local_affinity[~valid] = 0.0

    score = (
        0.60 * gaussian
        + 0.25 * rank_affinity
        + 0.15 * local_affinity
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
