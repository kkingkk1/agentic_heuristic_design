import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    undirected_d = 0.5 * (d + d.T)
    safe_d = np.where(valid, undirected_d, np.inf)

    positive = safe_d[valid & (safe_d > 0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    nearest = np.min(safe_d, axis=1)
    local_floor = max(0.25 * global_scale, np.finfo(float).eps)
    local_scale = np.where(
        np.isfinite(nearest),
        np.maximum(nearest, local_floor),
        global_scale,
    )

    temperature = max(0.70 * global_scale, np.finfo(float).eps)
    global_ratio = np.clip(safe_d / temperature, 0.0, 50.0)
    global_affinity = np.exp(-global_ratio)
    global_affinity[~valid] = 0.0

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    local_ratio = np.clip(
        safe_d / np.maximum(pair_scale, np.finfo(float).eps),
        0.0,
        50.0,
    )
    local_affinity = np.exp(-local_ratio)
    local_affinity[~valid] = 0.0

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=int)

    for i in range(n):
        row_order = order[i]
        row_order = row_order[valid[i, row_order]]
        ranks[i, row_order] = np.arange(row_order.size)

    candidate = valid & (ranks < 4)
    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    rank_score = np.exp(-0.70 * ranks.astype(float))
    rank_score[~valid] = 0.0

    enhanced = (
        0.54 * global_affinity
        + 0.26 * local_affinity
        + 0.16 * rank_score
        + 0.04 * mutual_nearest.astype(float)
    )
    enhanced[~candidate] = 0.0

    candidate_values = enhanced[candidate]
    reference = (
        float(np.median(candidate_values))
        if candidate_values.size
        else 1.0
    )
    exploration_prior = 0.08 * max(reference, np.finfo(float).eps)

    score = np.where(candidate, enhanced, exploration_prior * valid.astype(float))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
