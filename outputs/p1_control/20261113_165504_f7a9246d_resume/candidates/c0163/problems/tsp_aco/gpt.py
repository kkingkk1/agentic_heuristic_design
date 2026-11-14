import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, locally balanced priors for undirected TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    directed_valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return result

    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)

    order = np.argsort(pair_d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(pair_d, order, axis=1)

    ranks = np.full((n, n), n, dtype=int)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n)[None, :]
    ranks[~valid] = n

    nearest = sorted_d[:, 0]
    second = sorted_d[:, 1] if n >= 2 else np.full(n, np.inf)

    positive_edges = pair_d[valid & (pair_d > 0.0)]
    positive_nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]

    if positive_nearest.size:
        global_scale = float(np.median(positive_nearest))
    elif positive_edges.size:
        global_scale = float(np.median(positive_edges))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    neighbor_count = min(4, n - 1)
    local_values = sorted_d[:, :neighbor_count]
    local_values = local_values[np.isfinite(local_values) & (local_values > 0.0)]

    local_fallback = (
        float(np.median(local_values))
        if local_values.size
        else global_scale
    )

    local_scale = np.full(n, local_fallback, dtype=float)
    for i in range(n):
        values = sorted_d[i, :neighbor_count]
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            local_scale[i] = float(np.median(values))

    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    global_ratio = np.divide(
        pair_d,
        global_scale,
        out=np.full_like(pair_d, np.inf),
        where=valid,
    )
    local_ratio = np.divide(
        pair_d,
        pair_scale,
        out=np.full_like(pair_d, np.inf),
        where=valid,
    )

    global_score = np.exp(-np.minimum(global_ratio, 40.0))
    local_score = np.exp(-np.minimum(local_ratio, 40.0))

    rank_score = np.exp(-0.62 * np.minimum(ranks, n))
    rank_score[~valid] = 0.0

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    regret = np.zeros(n, dtype=float)
    has_two = (
        np.isfinite(nearest)
        & np.isfinite(second)
        & (nearest > 0.0)
        & (second > nearest)
    )
    regret[has_two] = np.clip(
        (second[has_two] - nearest[has_two])
        / np.maximum(second[has_two], eps),
        0.0,
        1.0,
    )

    regret_bonus = nearest_edge * (
        0.5 * (regret[:, None] + regret[None, :])
    )

    candidate_limit = min(4, n - 1)
    candidate = valid & (ranks <= candidate_limit)
    candidate_support = candidate | candidate.T

    score = (
        0.30 * global_score
        + 0.36 * local_score
        + 0.18 * rank_score
        + 0.08 * regret_bonus
        + 0.08 * mutual_nearest.astype(float)
    )

    score *= np.where(candidate_support, 1.0, 0.82)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
