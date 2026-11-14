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
    positive_values = d[positive_mask]
    fallback = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    eps = np.finfo(float).eps

    safe_distance = np.where(valid, np.maximum(d, 0.0), np.inf)

    sorted_distances = np.sort(safe_distance, axis=1)
    scale_k = min(3, n - 1)
    nearest = sorted_distances[:, :scale_k]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    normalized_distance = np.divide(
        safe_distance,
        endpoint_scale,
        out=np.full_like(safe_distance, np.inf, dtype=float),
        where=np.isfinite(safe_distance),
    )

    distance_affinity = np.exp(-normalized_distance)
    broad_affinity = np.exp(-0.5 * normalized_distance)

    row_order = np.argsort(safe_distance, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_weight = 1.0 / (1.0 + ranks)
    rank_weight[~valid] = 0.0
    row_rank_max = np.maximum(
        np.max(rank_weight, axis=1, keepdims=True),
        eps,
    )
    rank_weight /= row_rank_max

    reciprocal_rank = np.sqrt(rank_weight * rank_weight.T)

    top_k = min(3, n - 1)
    topk = np.zeros((n, n), dtype=bool)
    topk[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True
    topk &= valid

    mutual_topk = topk & topk.T
    mutual_top1 = mutual_topk & (ranks <= 0.0) & (ranks.T <= 0.0)

    reciprocal_bonus = np.ones((n, n), dtype=float)
    reciprocal_bonus += 0.40 * mutual_topk
    reciprocal_bonus += 0.30 * mutual_top1
    reciprocal_bonus *= 0.70 + 0.30 * reciprocal_rank

    endpoint_preference = np.maximum(
        rank_weight,
        rank_weight.T,
    )
    competition_factor = 0.80 + 0.20 * endpoint_preference

    score = (
        distance_affinity
        * (0.55 + 0.45 * broad_affinity)
        * (0.35 + 0.65 * reciprocal_rank)
        * reciprocal_bonus
        * competition_factor
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
