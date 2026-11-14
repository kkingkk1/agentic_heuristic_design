import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, globally normalized edge-priority indicators for TSP."""

    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid_directed = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid_directed, False)

    valid = valid_directed | valid_directed.T
    if not np.any(valid):
        return result

    symmetric = np.zeros_like(d, dtype=float)

    both_valid = valid_directed & valid_directed.T
    only_forward = valid_directed & ~valid_directed.T
    only_backward = ~valid_directed & valid_directed.T

    symmetric[both_valid] = (
        d[both_valid] + d.T[both_valid]
    ) / 2.0
    symmetric[only_forward] = d[only_forward]
    symmetric[only_backward] = d.T[only_backward]

    symmetric[~valid] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    finite_edges = symmetric[valid]
    global_scale = float(np.median(finite_edges))
    if not np.isfinite(global_scale) or global_scale <= 0:
        positive_edges = finite_edges[finite_edges > 0]
        global_scale = (
            float(np.median(positive_edges))
            if positive_edges.size
            else 1.0
        )

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    local_scales = np.empty(n, dtype=float)
    best = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)

    for i in range(n):
        row = symmetric[i]
        neighbors = row[np.isfinite(row)]
        if neighbors.size == 0:
            local_scales[i] = global_scale
            continue

        neighbors = np.sort(neighbors)
        best[i] = neighbors[0]
        if neighbors.size > 1:
            second[i] = neighbors[1]
        else:
            second[i] = neighbors[0]

        k = min(3, neighbors.size)
        local_scales[i] = float(np.median(neighbors[:k]))

    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0),
        local_scales,
        global_scale,
    )
    local_scales = np.maximum(local_scales, eps)

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)
    urgency = 1.0 + 0.7 * regret

    row_rank = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        indices = np.flatnonzero(np.isfinite(symmetric[i]))
        if indices.size:
            order = indices[np.argsort(symmetric[i, indices], kind="stable")]
            row_rank[i, order] = np.arange(order.size, dtype=float)

    local_quality = np.exp(
        -symmetric / np.sqrt(local_scales[:, None] * local_scales[None, :])
    )
    absolute_quality = np.exp(-symmetric / global_scale)

    rank_quality = 1.0 / (
        1.0
        + row_rank
        + row_rank.T
    )

    nearest_threshold = np.minimum(3, np.maximum(1, n - 1))
    mutual_neighbor = (
        (row_rank < nearest_threshold)
        & (row_rank.T < nearest_threshold)
    ).astype(float)
    mutual_bonus = 1.0 + 0.35 * mutual_neighbor

    endpoint_urgency = np.sqrt(
        urgency[:, None] * urgency[None, :]
    )

    score = (
        np.sqrt(local_quality * absolute_quality)
        * rank_quality
        * mutual_bonus
        * endpoint_urgency
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = (score + score.T) / 2.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0:
        result = score / maximum

    return result
