import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric stochastic edge-priority matrix for an undirected TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    safe = np.where(valid, d, np.inf)
    ordered = np.sort(safe, axis=1)

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    k = min(3, n - 1)
    local_values = ordered[:, :k]
    finite_local = local_values[np.isfinite(local_values)]
    positive_local = finite_local[finite_local > 0.0]
    fallback = float(np.median(positive_local)) if positive_local.size else global_scale

    local_scale = np.median(
        np.where(np.isfinite(local_values), local_values, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    global_affinity = np.exp(-np.minimum(safe / global_scale, 40.0))
    local_affinity = np.exp(-np.minimum(safe / pair_scale, 40.0))

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=int)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    rank_affinity = np.exp(-0.65 * ranks.astype(float))
    rank_affinity[~valid] = 0.0

    nearest = ordered[:, 0]
    second = ordered[:, min(1, n - 2)]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)
    second = np.where(np.isfinite(second), second, nearest)

    regret = np.maximum(second - nearest, 0.0)
    regret_scale = regret[regret > 0.0]
    regret_scale = (
        float(np.median(regret_scale))
        if regret_scale.size
        else global_scale
    )
    regret_scale = max(regret_scale, np.finfo(float).eps)

    endpoint_regret = 1.0 + np.minimum(regret / regret_scale, 3.0)
    regret_affinity = np.sqrt(
        endpoint_regret[:, None] * endpoint_regret[None, :]
    )

    nearest_mask = (ranks == 0) & valid
    top_three_mask = (ranks <= 2) & valid
    mutual_nearest = nearest_mask & nearest_mask.T
    mutual_top_three = top_three_mask & top_three_mask.T

    candidate_mask = (ranks <= min(3, n - 1)) | (
        ranks.T <= min(3, n - 1)
    )
    suppression = np.where(candidate_mask, 1.0, 0.82)

    score = (
        0.28 * global_affinity
        + 0.28 * local_affinity
        + 0.20 * rank_affinity
        + 0.10 * mutual_nearest.astype(float)
        + 0.06 * mutual_top_three.astype(float)
        + 0.08 * regret_affinity * local_affinity
    )

    score *= suppression
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
