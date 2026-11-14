import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, regret-aware stochastic prior for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    global_scale = max(global_scale, eps)

    safe = np.where(valid, d, np.inf)

    # Local scale estimated from the nearest positive incident edges.
    positive_safe = np.where(valid & (d > 0.0), d, np.inf)
    sorted_positive = np.sort(positive_safe, axis=1)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    local_scale = np.median(sorted_positive[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    # Global and endpoint-relative distance preference.
    local_closeness = np.exp(
        -np.clip(np.where(valid, d, 0.0) / pair_scale, 0.0, 60.0)
    )
    global_closeness = np.exp(
        -np.clip(np.where(valid, d, 0.0) / global_scale, 0.0, 60.0)
    )

    local_closeness = np.where(valid, local_closeness, 0.0)
    global_closeness = np.where(valid, global_closeness, 0.0)

    # Stable endpoint-relative ranks.
    order = np.argsort(safe, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    rank_score = np.where(valid, rank_score, 0.0)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    # Explicit reciprocal-nearest-neighbor support.
    nearest = np.argmin(safe, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = int(nearest[i])
        if valid[i, j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    # Endpoint regret: retaining a distinctive nearest edge is valuable.
    nearest_distance = np.min(safe, axis=1)
    second_distance = np.partition(safe, kth=min(1, n - 2), axis=1)[
        :, min(1, n - 2)
    ]

    nearest_distance = np.where(
        np.isfinite(nearest_distance),
        nearest_distance,
        global_scale,
    )
    second_distance = np.where(
        np.isfinite(second_distance),
        second_distance,
        nearest_distance,
    )

    regret_gap = np.maximum(second_distance - nearest_distance, 0.0)
    regret_scale = (
        float(np.median(regret_gap[regret_gap > 0.0]))
        if np.any(regret_gap > 0.0)
        else global_scale
    )
    regret_scale = max(regret_scale, eps)

    endpoint_regret = np.tanh(regret_gap / regret_scale)
    regret_support = np.sqrt(
        endpoint_regret[:, None] * endpoint_regret[None, :]
    )

    # Savings signal favors edges that jointly replace two nearest connections.
    savings = np.maximum(
        nearest_distance[:, None]
        + nearest_distance[None, :]
        - np.where(valid, d, np.inf),
        0.0,
    )
    savings = np.where(valid, savings, 0.0)

    positive_savings = savings[savings > 0.0]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else global_scale
    )
    savings_scale = max(savings_scale, eps)
    savings_score = np.tanh(savings / savings_scale)

    # Combine endpoint consensus, distance, rank, regret, and savings.
    consensus_closeness = np.sqrt(
        local_closeness * local_closeness.T
    )
    consensus_global = np.sqrt(
        global_closeness * global_closeness.T
    )

    score = (
        0.38 * consensus_closeness
        + 0.18 * consensus_global
        + 0.20 * mutual_rank
        + 0.10 * regret_support
        + 0.09 * savings_score
        + 0.05 * mutual_nearest
    )

    # Keep every valid edge available for stochastic exploration.
    score = np.where(valid, score, 0.0)
    score = 0.98 * score + 0.02 * valid.astype(float)

    # TSP edge priors are symmetric for undirected distance matrices.
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
