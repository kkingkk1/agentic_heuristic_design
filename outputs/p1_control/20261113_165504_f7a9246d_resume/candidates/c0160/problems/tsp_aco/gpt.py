import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, regret-aware stochastic priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    pair_distance = np.where(valid, 0.5 * (d + d.T), np.inf)
    eps = np.finfo(float).eps

    positive = pair_distance[valid & (pair_distance > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    safe = pair_distance
    order = np.argsort(safe, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(safe, order, axis=1)

    nearest = sorted_distance[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    if n > 2:
        second = sorted_distance[:, 1]
        second = np.where(np.isfinite(second), second, nearest + global_scale)
    else:
        second = nearest + global_scale

    local_scale = np.where(nearest > eps, nearest, global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    global_affinity = np.exp(
        -np.clip(pair_distance / (1.05 * global_scale), 0.0, 50.0)
    )
    local_affinity = np.exp(
        -np.clip(pair_distance / pair_scale, 0.0, 50.0)
    )

    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    rank_affinity = np.exp(-0.72 * np.minimum(ranks, n))
    rank_affinity = np.where(valid, rank_affinity, 0.0)

    relative_regret = np.divide(
        np.maximum(second - nearest, 0.0),
        np.maximum(second, eps),
    )
    relative_regret = np.clip(relative_regret, 0.0, 1.0)

    nearest_edge = valid & (ranks == 0)
    endpoint_regret = 0.5 * (
        relative_regret[:, None] + relative_regret[None, :]
    )
    regret_affinity = nearest_edge * endpoint_regret

    mutual_nearest = (
        nearest_edge & nearest_edge.T
    ).astype(float)

    local_opportunity = np.maximum(
        local_scale[:, None] + local_scale[None, :] - pair_distance,
        0.0,
    )
    local_opportunity /= np.maximum(
        local_scale[:, None] + local_scale[None, :],
        eps,
    )

    score = (
        0.30 * global_affinity
        + 0.34 * local_affinity
        + 0.16 * rank_affinity
        + 0.10 * regret_affinity
        + 0.07 * mutual_nearest
        + 0.03 * local_opportunity
    )

    implausibly_long = pair_distance > 2.5 * (
        local_scale[:, None] + local_scale[None, :]
    )
    score *= np.where(implausibly_long, 0.55, 1.0)

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
