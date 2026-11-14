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

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0

    safe_distance = np.where(valid, np.maximum(d, 0.0), np.inf)

    positive_distances = np.where(
        valid & (d > 0.0),
        d,
        np.inf,
    )
    sorted_distances = np.sort(positive_distances, axis=1)

    base_k = min(n - 1, max(2, int(np.ceil(np.sqrt(n)))))
    nearest = sorted_distances[:, :base_k]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    median_scale = max(float(np.median(local_scale)), eps)
    sparsity = np.clip(local_scale / median_scale, 0.65, 1.8)

    adaptive_k = np.rint(base_k * np.sqrt(sparsity)).astype(int)
    adaptive_k = np.clip(adaptive_k, 2, n - 1)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    normalized_distance = np.divide(
        safe_distance,
        endpoint_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(safe_distance),
    )

    rank_order = np.argsort(
        np.argsort(safe_distance, axis=1),
        axis=1,
    )
    rank = rank_order.astype(float) + 1.0

    reciprocal_rank = np.sqrt(
        (1.0 / rank) * (1.0 / rank.T)
    )
    reciprocal_rank /= np.maximum(
        np.max(reciprocal_rank, axis=1, keepdims=True),
        eps,
    )

    top_k = rank_order < adaptive_k[:, None]
    mutual_top_k = top_k & top_k.T

    local_affinity = np.exp(-0.85 * normalized_distance)
    rank_factor = 0.58 + 0.42 * reciprocal_rank
    neighborhood_factor = 1.0 + 0.18 * mutual_top_k.astype(float)

    score = local_affinity * rank_factor * neighborhood_factor
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
