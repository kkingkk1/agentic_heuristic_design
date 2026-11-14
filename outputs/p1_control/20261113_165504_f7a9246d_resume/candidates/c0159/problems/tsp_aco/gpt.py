import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic edge-priority indicators for a TSP distance matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(d) & (d >= 0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)
    positive = pair_d[valid & (pair_d > 0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(pair_d, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=float)
    row = np.arange(n)[:, None]
    sorted_distances = pair_d[row, order]

    finite_positions = np.isfinite(sorted_distances)
    ranks[row, order] = np.where(
        finite_positions,
        np.arange(n, dtype=float)[None, :],
        float(n),
    )
    ranks[~valid] = float(n)

    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n >= 2 else nearest
    third = sorted_distances[:, 2] if n >= 3 else second

    nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    second = np.where(np.isfinite(second), second, nearest)
    third = np.where(np.isfinite(third), third, second)

    local_scale = 0.55 * second + 0.30 * third + 0.15 * nearest
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    global_affinity = np.exp(
        -np.clip(pair_d / global_scale, 0.0, 40.0)
    )
    local_affinity = np.exp(
        -np.clip(pair_d / pair_scale, 0.0, 40.0)
    )

    rank_affinity = np.exp(-0.58 * np.minimum(ranks, float(n)))
    rank_affinity[~valid] = 0.0

    nearest_mask = ranks == 0
    mutual_nearest = nearest_mask & nearest_mask.T

    has_second = np.isfinite(sorted_distances[:, 1])
    regret = np.zeros(n, dtype=float)
    denominator = np.maximum(second, eps)
    regret[has_second] = np.clip(
        (second[has_second] - nearest[has_second])
        / denominator[has_second],
        0.0,
        1.0,
    )

    selective_regret = (
        nearest_mask
        * nearest_mask.T
        * np.sqrt(regret[:, None] * regret[None, :])
    )

    candidate_limit = min(4, n - 1)
    candidate = (ranks <= candidate_limit) | (ranks.T <= candidate_limit)

    nearest_sum = nearest[:, None] + nearest[None, :]
    bridge_value = np.maximum(nearest_sum - pair_d, 0.0)
    positive_bridge = bridge_value[valid & (bridge_value > 0)]
    bridge_scale = (
        float(np.median(positive_bridge))
        if positive_bridge.size
        else global_scale
    )
    bridge_scale = max(bridge_scale, eps)
    bridge_affinity = np.tanh(
        0.75 * np.clip(bridge_value / bridge_scale, 0.0, 20.0)
    )

    score = (
        0.38 * global_affinity
        + 0.30 * local_affinity
        + 0.18 * rank_affinity
        + 0.08 * mutual_nearest.astype(float)
        + 0.04 * selective_regret
        + 0.02 * bridge_affinity
    )

    score *= np.where(candidate, 1.0, 0.88)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
