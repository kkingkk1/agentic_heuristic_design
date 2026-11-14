import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return locally normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, eps)

    positive_edges = np.where(valid & (d > 0.0), d, np.inf)
    ordered_edges = np.sort(positive_edges, axis=1)

    k_scale = min(3, n - 1)
    nearest_edges = ordered_edges[:, :k_scale]
    nearest_edges = np.where(
        np.isfinite(nearest_edges),
        nearest_edges,
        fallback,
    )

    local_scale = np.maximum(np.mean(nearest_edges, axis=1), eps)
    endpoint_scale = np.maximum(
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        eps,
    )

    normalized_distance = np.divide(
        np.where(valid, np.maximum(d, 0.0), np.inf),
        endpoint_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    distance_affinity = np.exp(-np.minimum(normalized_distance, 60.0))
    distance_affinity /= 1.0 + normalized_distance

    rank_order = np.argsort(
        np.where(valid, normalized_distance, np.inf),
        axis=1,
    )
    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)

    for row in range(n):
        ranks[row, rank_order[row]] = rank_values

    rank_affinity = 1.0 / np.sqrt(np.maximum(ranks, 1.0))
    rank_affinity[~valid] = 0.0
    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)

    k_neighbor = min(4, n - 1)
    nearest_mask = np.zeros((n, n), dtype=bool)

    for row in range(n):
        candidates = rank_order[row]
        candidates = candidates[valid[row, candidates]]
        nearest_mask[row, candidates[:k_neighbor]] = True

    one_sided = nearest_mask | nearest_mask.T
    reciprocal = nearest_mask & nearest_mask.T
    neighborhood_bonus = (
        1.0
        + 0.30 * one_sided.astype(float)
        + 0.45 * reciprocal.astype(float)
    )

    mean_scale = max(float(np.mean(local_scale)), eps)
    sparsity = np.clip(local_scale / mean_scale, 0.25, 4.0)
    bridge_strength = np.sqrt(sparsity[:, None] * sparsity[None, :])
    bridge_bonus = 0.85 + 0.15 * np.clip(bridge_strength, 0.5, 2.0)

    score = (
        distance_affinity
        * (0.60 + 0.40 * mutual_rank)
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
