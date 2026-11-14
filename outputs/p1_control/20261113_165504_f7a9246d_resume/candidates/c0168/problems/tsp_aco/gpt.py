import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    symmetric_distance = 0.5 * (d + d.T)
    pair_valid = valid & valid.T
    symmetric_distance[~pair_valid] = d[~pair_valid]

    safe_d = np.where(valid, d, np.inf)
    sorted_d = np.sort(safe_d, axis=1)

    k = min(5, n - 1)
    nearest = sorted_d[:, :k]
    finite_nearest = np.isfinite(nearest)

    positive_nearest = nearest[(nearest > 0) & finite_nearest]
    positive_all = d[valid & (d > 0)]

    eps = np.finfo(float).eps

    if positive_all.size:
        global_scale = float(np.percentile(positive_all, 40.0))
    elif positive_nearest.size:
        global_scale = float(np.median(positive_nearest))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    local_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = nearest[i]
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            local_scale[i] = max(float(np.median(values)), eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    relative_local = np.divide(
        symmetric_distance,
        endpoint_scale,
        out=np.full_like(symmetric_distance, np.inf),
        where=endpoint_scale > 0,
    )
    relative_global = symmetric_distance / global_scale

    local_evidence = np.exp(-np.minimum(relative_local, 50.0))
    global_evidence = np.exp(-np.minimum(relative_global, 50.0))

    ranks = np.full((n, n), n, dtype=int)
    for i in range(n):
        order = np.argsort(safe_d[i], kind="stable")
        valid_order = order[valid[i, order]]
        ranks[i, valid_order] = np.arange(valid_order.size)

    rank_evidence = np.zeros_like(symmetric_distance, dtype=float)
    rank_evidence += (ranks == 0) * 1.00
    rank_evidence += (ranks == 1) * 0.68
    rank_evidence += (ranks == 2) * 0.42
    rank_evidence += (ranks == 3) * 0.22

    reciprocal_rank = np.where(
        ranks < n,
        1.0 / (1.0 + ranks.astype(float)),
        0.0,
    )
    reciprocal_rank = 0.5 * (reciprocal_rank + reciprocal_rank.T)

    mutual_nearest = (
        (ranks == 0) & (ranks.T == 0)
    ).astype(float)

    local_budget = local_scale[:, None] + local_scale[None, :]
    savings = np.divide(
        np.maximum(local_budget - symmetric_distance, 0.0),
        local_budget + eps,
    )

    zero_edge = (symmetric_distance == 0.0) & pair_valid

    endpoint_ratio = np.divide(
        symmetric_distance,
        np.maximum(
            np.minimum(local_scale[:, None], local_scale[None, :]),
            eps,
        ),
    )

    ratio_values = endpoint_ratio[pair_valid]
    if ratio_values.size:
        long_threshold = max(
            1.8,
            float(np.percentile(ratio_values, 85.0)),
        )
    else:
        long_threshold = 1.8

    unusually_long = (
        (symmetric_distance > long_threshold * local_scale[:, None])
        & (symmetric_distance > long_threshold * local_scale[None, :])
    )

    score = (
        0.28 * local_evidence
        + 0.22 * global_evidence
        + 0.20 * rank_evidence
        + 0.14 * reciprocal_rank
        + 0.10 * mutual_nearest
        + 0.06 * savings
    )

    score *= np.where(unusually_long, 0.50, 1.0)

    score = np.where(zero_edge, np.maximum(score, 1.0), score)
    score[~pair_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
