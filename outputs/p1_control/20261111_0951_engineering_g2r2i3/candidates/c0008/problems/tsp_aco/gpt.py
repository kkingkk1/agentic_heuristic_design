import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    positive_rows = np.where(valid & (d > 0.0), d, np.inf)
    k_scale = min(3, n - 1)
    nearest_positive = np.sort(positive_rows, axis=1)[:, :k_scale]
    nearest_positive = np.where(
        np.isfinite(nearest_positive),
        nearest_positive,
        fallback,
    )
    local_scale = np.mean(nearest_positive, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    safe_distance = np.where(valid, d, np.inf)
    normalized_distance = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    distance_affinity = np.exp(-np.minimum(normalized_distance, 60.0))
    distance_affinity *= 1.0 / (1.0 + normalized_distance)

    rank_order = np.argsort(
        np.where(valid, normalized_distance, np.inf),
        axis=1,
    )
    ranks = np.empty_like(rank_order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    for row in range(n):
        ranks[row, rank_order[row]] = rank_values

    rank_affinity = 1.0 / np.sqrt(np.maximum(ranks, 1.0))
    rank_affinity[~valid] = 0.0

    k_neighbor = min(4, n - 1)
    nearest_mask = np.zeros((n, n), dtype=bool)
    for row in range(n):
        candidates = rank_order[row]
        candidates = candidates[valid[row, candidates]]
        nearest_mask[row, candidates[:k_neighbor]] = True

    reciprocal = nearest_mask & nearest_mask.T
    one_sided = nearest_mask | nearest_mask.T
    neighborhood_bonus = (
        1.0
        + 0.35 * one_sided.astype(float)
        + 0.45 * reciprocal.astype(float)
    )

    mean_scale = max(
        float(np.mean(local_scale)),
        np.finfo(float).eps,
    )
    relative_sparsity = np.clip(local_scale / mean_scale, 0.25, 4.0)
    bridge_bonus = np.sqrt(
        relative_sparsity[:, None] * relative_sparsity[None, :]
    )
    bridge_bonus = 0.8 + 0.2 * np.clip(bridge_bonus, 0.25, 2.5)

    score = (
        distance_affinity
        * (0.65 + 0.35 * rank_affinity)
        * neighborhood_bonus
        * bridge_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
