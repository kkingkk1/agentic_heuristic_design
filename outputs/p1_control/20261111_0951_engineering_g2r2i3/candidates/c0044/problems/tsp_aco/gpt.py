import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    pair_distance = 0.5 * (d + d.T)
    positive = pair_distance[valid & (pair_distance > 0.0)]
    fallback = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    positive_rows = np.where(
        valid & (pair_distance > 0.0),
        pair_distance,
        np.inf,
    )
    k_scale = min(3, n - 1)
    nearest = np.sort(positive_rows, axis=1)[:, :k_scale]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    safe_distance = np.where(valid, pair_distance, np.inf)
    normalized_distance = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    distance_affinity = np.exp(-np.minimum(normalized_distance, 60.0))
    distance_affinity /= 1.0 + normalized_distance

    rank_order = np.argsort(normalized_distance, axis=1)
    ranks = np.empty_like(rank_order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)

    for row in range(n):
        ranks[row, rank_order[row]] = rank_values

    ranks[~valid] = np.inf

    reciprocal_rank = np.zeros_like(normalized_distance, dtype=float)
    reciprocal_rank[valid] = 1.0 / np.sqrt(
        ranks[valid] * ranks.T[valid]
    )

    mean_scale = max(
        float(np.mean(local_scale)),
        np.finfo(float).eps,
    )
    relative_sparsity = np.clip(local_scale / mean_scale, 0.25, 4.0)

    max_neighbors = n - 1
    adaptive_k = 2.0 + np.sqrt(relative_sparsity)
    adaptive_k = np.clip(
        np.rint(adaptive_k),
        1.0,
        float(max_neighbors),
    ).astype(int)

    top_k = valid & (ranks <= adaptive_k[:, None])
    one_sided = top_k | top_k.T
    reciprocal = top_k & top_k.T

    neighborhood_factor = (
        1.0
        + 0.20 * one_sided.astype(float)
        + 0.40 * reciprocal.astype(float)
    )

    bridge_factor = 0.90 + 0.10 * np.sqrt(
        np.clip(
            relative_sparsity[:, None] * relative_sparsity[None, :],
            0.25,
            4.0,
        )
    )

    score = (
        distance_affinity
        * (0.60 + 0.40 * reciprocal_rank)
        * neighborhood_factor
        * bridge_factor
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
