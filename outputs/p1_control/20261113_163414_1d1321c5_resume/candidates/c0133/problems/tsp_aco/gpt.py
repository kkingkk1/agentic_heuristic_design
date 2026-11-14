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
    finite_nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    fallback_scale = (
        float(np.median(finite_nearest))
        if finite_nearest.size
        else global_scale
    )

    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        fallback_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    temperature = max(0.70 * global_scale, np.finfo(float).eps)
    clipped = np.minimum(np.maximum(safe_d / temperature, 0.0), 50.0)
    global_affinity = np.exp(-clipped)
    global_affinity[~valid] = 0.0

    local_pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    local_ratio = np.minimum(
        np.maximum(undirected_d / local_pair_scale, 0.0),
        50.0,
    )
    local_affinity = np.exp(-local_ratio)
    local_affinity[~valid] = 0.0

    balanced = global_affinity.copy()
    for _ in range(8):
        row_sum = balanced.sum(axis=1)
        row_normalized = np.divide(
            balanced,
            row_sum[:, None],
            out=np.zeros_like(balanced),
            where=row_sum[:, None] > 0,
        )

        col_sum = row_normalized.sum(axis=0)
        col_normalized = np.divide(
            row_normalized,
            col_sum[None, :],
            out=np.zeros_like(balanced),
            where=col_sum[None, :] > 0,
        )

        balanced = 0.5 * balanced + 0.5 * col_normalized

    balanced = 0.5 * (balanced + balanced.T)
    balanced_max = float(np.max(balanced))
    if balanced_max > 0:
        balanced /= balanced_max

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=int)

    for i in range(n):
        row_indices = order[i]
        row_indices = row_indices[valid[i, row_indices]]
        ranks[i, row_indices] = np.arange(row_indices.size)

    rank_score = np.exp(-0.62 * ranks.astype(float))
    rank_score[~valid] = 0.0

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    second_distance = np.full(n, np.inf, dtype=float)
    for i in range(n):
        row = safe_d[i, valid[i]]
        if row.size >= 2:
            second_distance[i] = np.partition(row, 1)[1]

    regret = np.maximum(second_distance - nearest, 0.0)
    regret_scale = np.where(
        np.isfinite(second_distance) & (second_distance > 0),
        second_distance,
        global_scale,
    )
    regret_ratio = np.divide(
        regret,
        regret_scale,
        out=np.zeros_like(regret),
        where=regret_scale > 0,
    )

    endpoint_bonus = 1.0 + 0.10 * np.minimum(regret_ratio, 2.0)
    regret_weight = np.sqrt(
        endpoint_bonus[:, None] * endpoint_bonus[None, :]
    )
    regret_weight = np.where(
        nearest_edge | nearest_edge.T,
        regret_weight,
        1.0,
    )

    score = (
        0.40 * global_affinity
        + 0.28 * local_affinity
        + 0.20 * balanced
        + 0.10 * rank_score
        + 0.02 * mutual_nearest.astype(float)
    )

    score *= regret_weight

    exploration = valid & (ranks > 3)
    score = np.where(exploration, 0.88 * score, score)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
