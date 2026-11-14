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

    nearest_distance = np.min(safe_d, axis=1)
    finite_nearest = nearest_distance[
        np.isfinite(nearest_distance) & (nearest_distance > 0)
    ]
    fallback_scale = (
        float(np.median(finite_nearest))
        if finite_nearest.size
        else global_scale
    )
    local_scale = np.where(
        np.isfinite(nearest_distance) & (nearest_distance > 0),
        nearest_distance,
        fallback_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    k = min(4, n - 1)
    top_indices = np.argpartition(safe_d, kth=k - 1, axis=1)[:, :k]

    rows = np.arange(n)[:, None]
    top_distances = safe_d[rows, top_indices]
    top_valid = valid[rows, top_indices]

    denominator = np.sqrt(
        local_scale[:, None] * local_scale[top_indices]
    )
    denominator = np.maximum(denominator, np.finfo(float).eps)

    global_ratio = np.divide(
        top_distances,
        0.70 * global_scale,
        out=np.full_like(top_distances, np.inf),
        where=top_valid,
    )
    local_ratio = np.divide(
        top_distances,
        denominator,
        out=np.full_like(top_distances, np.inf),
        where=top_valid,
    )

    global_affinity = np.zeros((n, k), dtype=float)
    local_affinity = np.zeros((n, k), dtype=float)

    finite_positions = top_valid & np.isfinite(top_distances)
    global_affinity[finite_positions] = np.exp(
        -np.minimum(global_ratio[finite_positions], 50.0)
    )
    local_affinity[finite_positions] = np.exp(
        -np.minimum(local_ratio[finite_positions], 50.0)
    )

    balanced = global_affinity.copy()
    for _ in range(3):
        row_sum = balanced.sum(axis=1)
        row_normalized = np.divide(
            balanced,
            row_sum[:, None],
            out=np.zeros_like(balanced),
            where=row_sum[:, None] > 0,
        )

        col_sum = np.zeros(n, dtype=float)
        np.add.at(col_sum, top_indices.ravel(), row_normalized.ravel())

        col_normalized = np.divide(
            row_normalized,
            col_sum[top_indices],
            out=np.zeros_like(balanced),
            where=col_sum[top_indices] > 0,
        )
        balanced = 0.5 * balanced + 0.5 * col_normalized

    balanced_max = float(np.max(balanced))
    if balanced_max > 0:
        balanced /= balanced_max

    order = np.argsort(top_distances, axis=1, kind="stable")
    rank_weights = np.exp(-0.62 * np.arange(k, dtype=float))
    rank_score = np.broadcast_to(rank_weights, (n, k)).copy()

    nearest_indices = np.argmin(safe_d, axis=1)
    nearest_edge = (
        top_valid
        & (top_indices == nearest_indices[:, None])
    )
    mutual_nearest = nearest_edge & (
        nearest_indices[top_indices] == np.arange(n)[:, None]
    )

    second_distance = np.full(n, np.inf, dtype=float)
    for i in range(n):
        row = safe_d[i, valid[i]]
        if row.size >= 2:
            second_distance[i] = np.partition(row, 1)[1]

    regret = np.maximum(second_distance - nearest_distance, 0.0)
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
        endpoint_bonus[:, None] * endpoint_bonus[top_indices]
    )
    regret_weight = np.where(
        nearest_edge | nearest_edge.T[top_indices],
        regret_weight,
        1.0,
    )

    candidate_score = (
        0.38 * global_affinity
        + 0.27 * local_affinity
        + 0.20 * balanced
        + 0.13 * rank_score
        + 0.02 * mutual_nearest.astype(float)
    )
    candidate_score *= regret_weight
    candidate_score[~top_valid] = 0.0

    score = np.full((n, n), 0.015, dtype=float)
    score[~valid] = 0.0
    score[rows, top_indices] = np.maximum(
        score[rows, top_indices],
        candidate_score,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
