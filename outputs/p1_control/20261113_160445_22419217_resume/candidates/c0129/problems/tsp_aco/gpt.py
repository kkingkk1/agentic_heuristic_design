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

    # TSP is undirected here; retain only mutually valid edges.
    valid = valid & valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    undirected_d = 0.5 * (d + d.T)
    safe_d = np.where(valid, undirected_d, np.inf)

    positive = safe_d[valid & (safe_d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    nearest = np.min(safe_d, axis=1)
    finite_nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        float(np.median(finite_nearest)) if finite_nearest.size else scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Temperature-scaled global affinity.
    temperature = max(0.75 * scale, np.finfo(float).eps)
    affinity = np.exp(-np.minimum(np.maximum(safe_d / temperature, 0.0), 50.0))
    affinity[~valid] = 0.0

    # Sinkhorn-like row-column balancing on the valid affinity matrix.
    balanced = affinity.copy()
    for _ in range(28):
        row_sum = balanced.sum(axis=1)
        balanced = np.divide(
            balanced,
            row_sum[:, None],
            out=np.zeros_like(balanced),
            where=row_sum[:, None] > 0,
        )
        col_sum = balanced.sum(axis=0)
        balanced = np.divide(
            balanced,
            col_sum[None, :],
            out=np.zeros_like(balanced),
            where=col_sum[None, :] > 0,
        )

    balanced = 0.5 * (balanced + balanced.T)
    balanced_max = float(np.max(balanced))
    if balanced_max > 0:
        balanced /= balanced_max

    # Stable local ranks, excluding invalid edges and the diagonal.
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=int)
    for i in range(n):
        row = order[i]
        row = row[valid[i, row]]
        ranks[i, row] = np.arange(row.size)

    rank_score = np.exp(-0.68 * ranks.astype(float))
    rank_score[~valid] = 0.0

    nearest_edge = (ranks == 0) & valid
    second_distance = np.full(n, np.inf, dtype=float)
    for i in range(n):
        row = safe_d[i, valid[i]]
        if row.size >= 2:
            second_distance[i] = np.partition(row, 1)[1]

    regret = np.maximum(second_distance - nearest, 0.0)
    regret_scale = np.where(
        np.isfinite(second_distance) & (second_distance > 0),
        second_distance,
        scale,
    )
    regret_bonus = np.divide(
        regret,
        regret_scale,
        out=np.zeros_like(regret),
        where=regret_scale > 0,
    )
    endpoint_regret = 1.0 + 0.22 * np.minimum(regret_bonus, 3.0)
    regret_weight = np.sqrt(endpoint_regret[:, None] * endpoint_regret[None, :])
    regret_weight = np.where(nearest_edge | nearest_edge.T, regret_weight, 1.0)

    mutual_nearest = nearest_edge & nearest_edge.T

    local_scale_pair = np.sqrt(local_scale[:, None] * local_scale[None, :])
    local_affinity = np.exp(
        -np.minimum(np.maximum(undirected_d / local_scale_pair, 0.0), 50.0)
    )
    local_affinity[~valid] = 0.0

    score = (
        0.42 * balanced
        + 0.30 * local_affinity
        + 0.20 * rank_score
        + 0.08 * mutual_nearest.astype(float)
    )
    score *= regret_weight

    # Retain weak exploration mass outside the first few local neighbors.
    score = np.where(valid & (ranks > 3), 0.84 * score, score)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
