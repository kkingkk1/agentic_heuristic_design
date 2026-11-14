import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structurally informed edge-priority indicators for stochastic TSP sampling."""
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
    fallback = max(fallback, eps)

    positive_distance = np.where(valid & (d > 0.0), d, np.inf)
    row_scale = np.min(positive_distance, axis=1, keepdims=True)
    row_scale = np.where(np.isfinite(row_scale), row_scale, fallback)
    row_scale = np.maximum(row_scale, eps)

    safe_distance = np.where(valid, np.maximum(d, 0.0), np.inf)
    normalized_distance = safe_distance / row_scale

    distance_affinity = 1.0 / (1.0 + normalized_distance)
    distance_affinity *= distance_affinity

    row_order = np.argsort(safe_distance, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_weight = 1.0 / (1.0 + ranks)
    rank_weight[~valid] = 0.0

    reciprocal_rank = np.sqrt(rank_weight * rank_weight.T)
    rank_factor = 0.40 + 0.60 * reciprocal_rank

    top_k = min(3, n - 1)
    topk = np.zeros((n, n), dtype=bool)
    topk[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True
    topk &= valid

    mutual_topk = topk & topk.T
    mutual_top1 = mutual_topk & (ranks == 0.0) & (ranks.T == 0.0)

    structural_bonus = (
        1.0
        + 0.25 * mutual_topk
        + 0.15 * mutual_top1
    )

    score = distance_affinity * rank_factor * structural_bonus
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    return np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    ).astype(float, copy=False)
