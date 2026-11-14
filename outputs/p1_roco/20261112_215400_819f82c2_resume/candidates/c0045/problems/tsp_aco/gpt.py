import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric nonnegative edge-prior indicators for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square two-dimensional array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    safe = np.where(finite, d, fallback)
    safe = np.maximum(safe, 0.0)
    np.fill_diagonal(safe, np.inf)

    off_diagonal = safe[np.isfinite(safe)]
    positive_off_diagonal = off_diagonal[off_diagonal > 0]
    global_scale = (
        float(np.median(positive_off_diagonal))
        if positive_off_diagonal.size
        else fallback
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    k = min(n - 1, max(2, int(np.sqrt(n))))
    ordered = np.sort(safe, axis=1)
    local_scale = ordered[:, :k]
    local_scale = np.median(local_scale, axis=1)
    local_scale[~np.isfinite(local_scale) | (local_scale <= 0)] = global_scale

    pair_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    valid = np.isfinite(safe)

    relative_shortness = np.zeros((n, n), dtype=float)
    relative_shortness[valid] = np.exp(
        -safe[valid] / pair_scale[valid]
    )

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        order = np.argsort(safe[i], kind="stable")
        rank = 0
        for j in order:
            if i != j and np.isfinite(safe[i, j]):
                ranks[i, j] = rank
                rank += 1

    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~valid] = 0.0
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    nearest_order = np.argsort(safe, axis=1, kind="stable")
    nearest = nearest_order[:, 0]

    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if i != j and j < n and nearest[j] == i and valid[i, j]:
            mutual_bonus[i, j] = 1.0
            mutual_bonus[j, i] = 1.0

    candidate_mask = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidate_mask[i, nearest_order[i, :k]] = 1.0
    np.fill_diagonal(candidate_mask, 0.0)

    candidate_mass = candidate_mask.sum(axis=1)
    edge_support = candidate_mask + candidate_mask.T
    scarcity = 1.0 / np.sqrt(
        (candidate_mass[:, None] + 1.0)
        * (candidate_mass[None, :] + 1.0)
    )
    scarcity = scarcity / np.max(scarcity)

    global_distance = np.zeros((n, n), dtype=float)
    global_distance[valid] = 1.0 / (
        1.0 + safe[valid] / global_scale
    )

    score = (
        0.42 * relative_shortness
        + 0.28 * rank_consensus
        + 0.15 * global_distance
        + 0.10 * scarcity * edge_support
        + 0.05 * mutual_bonus
    )

    score *= 0.75 + 0.25 * np.sqrt(
        np.maximum(rank_score, 0.0)
        * np.maximum(rank_score.T, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
