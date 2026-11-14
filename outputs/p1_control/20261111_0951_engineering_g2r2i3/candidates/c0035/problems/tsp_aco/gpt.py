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

    positive_mask = valid & (d > 0.0)
    positive = d[positive_mask]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    positive_distances = np.where(positive_mask, d, np.inf)
    sorted_distances = np.sort(positive_distances, axis=1)

    k_scale = min(3, n - 1)
    nearest = sorted_distances[:, :k_scale]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(valid, np.maximum(d, 0.0), np.inf)
    endpoint_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    endpoint_scale = np.maximum(endpoint_scale, eps)

    normalized_distance = np.divide(
        safe_distance,
        endpoint_scale,
        out=np.full_like(safe_distance, np.inf, dtype=float),
        where=np.isfinite(safe_distance),
    )

    distance_affinity = np.exp(-normalized_distance)
    soft_affinity = np.exp(-0.5 * normalized_distance)

    row_order = np.argsort(safe_distance, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    rank_weight = 1.0 / (1.0 + row_rank)
    rank_weight[~valid] = 0.0
    rank_weight = rank_weight / np.maximum(
        np.max(rank_weight, axis=1, keepdims=True), eps
    )

    reciprocal_rank = np.sqrt(rank_weight * rank_weight.T)

    k_reciprocal = min(3, n - 1)
    topk = np.zeros((n, n), dtype=bool)
    topk[np.arange(n)[:, None], row_order[:, :k_reciprocal]] = True
    topk &= valid
    mutual_topk = topk & topk.T

    reciprocal_bonus = np.where(mutual_topk, 1.35, 1.0)
    reciprocal_bonus *= 0.75 + 0.25 * reciprocal_rank

    median_scale = max(float(np.median(local_scale)), eps)
    sparsity = local_scale / median_scale
    sparse_bonus = np.sqrt(sparsity[:, None] * sparsity[None, :])
    sparse_bonus = np.clip(sparse_bonus, 0.75, 1.35)

    score = (
        distance_affinity
        * (0.50 + 0.50 * soft_affinity)
        * (0.45 + 0.55 * reciprocal_rank)
        * reciprocal_bonus
        * (0.90 + 0.10 * sparse_bonus)
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
