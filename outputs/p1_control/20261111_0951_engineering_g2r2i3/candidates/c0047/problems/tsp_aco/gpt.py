import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    pair_valid = valid | valid.T
    pair_distance = np.full((n, n), np.inf, dtype=float)

    both_valid = valid & valid.T
    pair_distance[both_valid] = 0.5 * (
        d[both_valid] + d.T[both_valid]
    )

    only_forward = valid & ~valid.T
    pair_distance[only_forward] = d[only_forward]

    only_reverse = valid.T & ~valid
    pair_distance[only_reverse] = d.T[only_reverse]

    np.fill_diagonal(pair_distance, np.inf)

    degree = np.sum(pair_valid, axis=1).astype(int)
    neighborhood_size = np.minimum(
        degree,
        np.maximum(2, np.ceil(np.sqrt(np.maximum(degree, 1))).astype(int)),
    )

    positive_pair_distance = np.where(
        pair_valid & (pair_distance > 0.0),
        pair_distance,
        np.inf,
    )
    sorted_positive = np.sort(positive_pair_distance, axis=1)

    local_scale = np.empty(n, dtype=float)
    for i in range(n):
        k = int(neighborhood_size[i])
        nearest = sorted_positive[i, :k]
        finite_nearest = nearest[np.isfinite(nearest)]
        local_scale[i] = (
            float(np.mean(finite_nearest))
            if finite_nearest.size
            else fallback
        )

    local_scale = np.maximum(local_scale, eps)
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    normalized_distance = np.divide(
        pair_distance,
        endpoint_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(pair_distance),
    )

    distance_affinity = np.exp(-normalized_distance)

    order = np.argsort(pair_distance, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    base_ranks = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = base_ranks

    reciprocal_rank = 1.0 / (1.0 + ranks)
    reciprocal_rank = np.sqrt(
        reciprocal_rank * reciprocal_rank.T
    )

    adaptive_k = np.maximum(
        1.0,
        np.minimum(
            5.0,
            np.sqrt(np.maximum(degree.astype(float), 1.0)),
        ),
    )

    excess_rank = np.maximum(
        ranks - adaptive_k[:, None] + 1.0,
        0.0,
    )
    topk_membership = 1.0 / (
        1.0 + excess_rank / adaptive_k[:, None]
    )
    mutual_topk = np.sqrt(
        topk_membership * topk_membership.T
    )

    score = (
        distance_affinity
        * (0.55 + 0.45 * reciprocal_rank)
        * (0.65 + 0.35 * mutual_topk)
    )

    score[~pair_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score = score / maximum

    return score.astype(float, copy=False)
