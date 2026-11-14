import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric stochastic edge-priority indicators for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return result

    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)
    eps = np.finfo(float).eps

    positive = pair_d[valid & (pair_d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(pair_d, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=int)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n)[None, :]
    ranks[~valid] = n

    nearest = np.min(pair_d, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    candidate_count = min(5, n - 1)
    neighborhood = np.take_along_axis(
        pair_d,
        order[:, :candidate_count],
        axis=1,
    )
    neighborhood = neighborhood[np.isfinite(neighborhood)]

    positive_neighborhood = neighborhood[neighborhood > 0.0]
    if positive_neighborhood.size:
        local_default = float(np.median(positive_neighborhood))
    else:
        local_default = global_scale

    local_scale = np.take_along_axis(
        pair_d,
        order[:, :candidate_count],
        axis=1,
    )
    local_scale = np.nanmedian(
        np.where(np.isfinite(local_scale), local_scale, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        local_default,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    global_affinity = np.exp(
        -np.clip(pair_d / global_scale, 0.0, 50.0)
    )
    local_affinity = np.exp(
        -np.clip(pair_d / endpoint_scale, 0.0, 50.0)
    )

    rank_affinity = np.exp(
        -0.55 * (
            ranks.astype(float)[:, :] + ranks.astype(float).T
        )
    )
    rank_affinity = np.where(valid, rank_affinity, 0.0)

    nearest_values = np.take_along_axis(
        pair_d,
        order[:, : min(2, n - 1)],
        axis=1,
    )
    nearest_value = nearest_values[:, 0]
    if nearest_values.shape[1] > 1:
        second_value = nearest_values[:, 1]
    else:
        second_value = nearest_value

    regret = np.zeros(n, dtype=float)
    has_alternatives = (
        np.isfinite(nearest_value)
        & np.isfinite(second_value)
        & (second_value > nearest_value)
    )
    regret[has_alternatives] = np.clip(
        (second_value[has_alternatives] - nearest_value[has_alternatives])
        / np.maximum(second_value[has_alternatives], eps),
        0.0,
        1.0,
    )

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    regret_bonus = (
        nearest_edge
        * 0.5
        * (regret[:, None] + regret[None, :])
    )

    candidate = valid & (ranks <= candidate_count)
    endpoint_long = (
        (pair_d > 2.4 * local_scale[:, None])
        & (pair_d > 2.4 * local_scale[None, :])
    )
    long_penalty = np.where(endpoint_long, 0.55, 1.0)

    score = (
        0.30 * global_affinity
        + 0.32 * local_affinity
        + 0.18 * rank_affinity
        + 0.10 * mutual_nearest.astype(float)
        + 0.10 * regret_bonus
    )

    score = np.where(candidate, score, 0.82 * score)
    score *= long_penalty
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~valid] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
