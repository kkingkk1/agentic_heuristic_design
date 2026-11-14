import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        global_scale = float(np.median(positive))
    elif values.size:
        global_scale = float(np.max(values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(values))
    penalty = max(finite_max, global_scale) * 8.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_distances = np.sort(work, axis=1)
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else nearest

    local_scale = 0.5 * (nearest + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = np.divide(
        work,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=pair_scale > 0.0,
    )

    inverse_distance = np.exp(
        -np.clip(1.18 * relative, 0.0, 60.0)
    )

    rank_scale = max(1.0, 0.72 * np.sqrt(float(n)))
    rank_strength = np.exp(-ranks / rank_scale)
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.38
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - nearest,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.36 * (
        regret[:, None] + regret[None, :]
    )

    nearest_node = order[:, 0]
    index = np.arange(n)

    mutual_nearest = (
        (nearest_node[:, None] == index[None, :])
        & (nearest_node[None, :] == index[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.30, 1.0)

    long_edge_penalty = np.exp(
        -0.88 * np.maximum(relative - 1.0, 0.0)
    )

    close = work <= 1.65 * local_scale[:, None]
    close[~valid] = False
    np.fill_diagonal(close, False)

    degree = np.sum(close, axis=1).astype(float)
    positive_degree = degree[degree > 0.0]
    degree_reference = (
        float(np.median(positive_degree))
        if positive_degree.size
        else 1.0
    )

    if not np.isfinite(degree_reference) or degree_reference <= 0.0:
        degree_reference = 1.0

    scarcity = np.sqrt(
        degree_reference / np.maximum(degree, 1.0)
    )
    scarcity = np.clip(scarcity, 0.86, 1.16)

    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    # Triangle detour advantage:
    # an edge is useful when no third node offers a much cheaper two-edge route.
    via_cost = work[:, :, None] + work[None, :, :]
    direct_cost = work[:, None, :]
    detour_ratio = np.divide(
        via_cost,
        np.maximum(direct_cost, eps),
    )

    detour_ratio[~np.isfinite(detour_ratio)] = np.inf
    np.fill_diagonal(detour_ratio, np.inf)

    best_detour = np.min(detour_ratio, axis=1)
    detour_advantage = np.clip(best_detour - 1.0, 0.0, 3.0)

    detour_term = 1.0 + 0.10 * np.minimum(
        detour_advantage,
        detour_advantage.T,
    )

    # Penalize edges that are simultaneously much worse than both endpoints'
    # local alternatives, but preserve a small allowance for sparse regions.
    endpoint_ratio = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    endpoint_penalty = np.exp(
        -0.18 * np.maximum(endpoint_ratio - 2.0, 0.0)
    )

    scores = (
        inverse_distance
        * (
            0.50
            + 0.28 * bidirectional_rank
            + 0.22 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * scarcity_term
        * detour_term
        * endpoint_penalty
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
