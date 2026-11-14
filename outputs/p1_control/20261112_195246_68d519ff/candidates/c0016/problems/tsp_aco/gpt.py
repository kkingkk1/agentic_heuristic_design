import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    safe_d = np.where(positive, d, np.inf)
    np.fill_diagonal(safe_d, np.inf)

    def robust_scale(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float)
        valid = np.isfinite(y)
        if not np.any(valid):
            return np.zeros_like(y)
        values = y[valid]
        lo = np.percentile(values, 10.0)
        hi = np.percentile(values, 90.0)
        if hi <= lo:
            lo = np.min(values)
            hi = np.max(values)
        if hi <= lo:
            return np.where(valid, 1.0, 0.0)
        z = (y - lo) / (hi - lo)
        return np.clip(np.where(valid, z, 0.0), 0.0, 1.0)

    finite_values = d[positive]
    distance_scale = np.median(finite_values)
    if not np.isfinite(distance_scale) or distance_scale <= 0:
        distance_scale = 1.0

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[positive] = 1.0 / (d[positive] + distance_scale * 1e-6)
    inverse_distance = robust_scale(inverse_distance)

    nearest_order = np.argsort(safe_d, axis=1)
    nearest_rank = np.empty((n, n), dtype=float)
    nearest_rank.fill(n)
    ranks = np.arange(n, dtype=float)
    for i in range(n):
        nearest_rank[i, nearest_order[i]] = ranks

    rank_score = np.zeros((n, n), dtype=float)
    valid_rank = np.isfinite(d) & ~np.eye(n, dtype=bool)
    rank_score[valid_rank] = np.exp(
        -nearest_rank[valid_rank] / max(1.0, np.sqrt(n))
    )
    rank_score = robust_scale(rank_score)

    mutual_support = np.zeros((n, n), dtype=float)
    for i in range(n):
        k_i = min(4, max(1, n - 1))
        neighbors_i = nearest_order[i, :k_i]
        for j in neighbors_i:
            if j != i and i in nearest_order[j, :k_i]:
                mutual_support[i, j] = 1.0
    mutual_support = np.maximum(mutual_support, mutual_support.T)

    local_pair_cost = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not (np.isfinite(safe_d[i, j]) and np.isfinite(safe_d[j, i])):
                continue
            alternatives_i = nearest_order[i]
            alternatives_j = nearest_order[j]
            ni = next((x for x in alternatives_i if x != i and x != j), None)
            nj = next((x for x in alternatives_j if x != i and x != j), None)
            if ni is None or nj is None:
                transition = 0.0
            else:
                transition = 0.5 * (
                    safe_d[i, ni] + safe_d[j, nj]
                )
            local_pair_cost[i, j] = safe_d[i, j] + transition
            local_pair_cost[j, i] = local_pair_cost[i, j]

    compatibility = np.zeros((n, n), dtype=float)
    valid_pair = np.isfinite(local_pair_cost)
    compatibility[valid_pair] = 1.0 / (
        local_pair_cost[valid_pair] + distance_scale * 1e-6
    )
    compatibility = robust_scale(compatibility)

    scores = (
        0.45 * inverse_distance
        + 0.25 * rank_score
        + 0.20 * compatibility
        + 0.10 * mutual_support
    )

    scores = 0.5 * (scores + scores.T)
    scores[~(np.isfinite(d) & ~np.eye(n, dtype=bool))] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores.astype(float, copy=False)
