import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.percentile(positive_values, 50.0))
        global_low = float(np.percentile(positive_values, 25.0))
        global_high = float(np.percentile(positive_values, 75.0))
    else:
        global_scale = 1.0
        global_low = 1.0
        global_high = 1.0

    global_scale = max(global_scale, eps)
    global_spread = max(global_high - global_low, eps)

    safe_d = np.where(valid, d, np.inf)
    sorted_d = np.sort(safe_d, axis=1)

    degree = valid.sum(axis=1)
    k = min(8, max(1, n - 1))
    k = min(k, max(1, int(np.max(degree))))

    neighborhood = sorted_d[:, :k]
    neighborhood_valid = np.isfinite(neighborhood)

    local_median = np.full(n, global_scale, dtype=float)
    local_q75 = np.full(n, global_high, dtype=float)

    for i in range(n):
        values = neighborhood[i][neighborhood_valid[i]]
        positive = values[values > 0.0]

        if values.size:
            local_median[i] = max(float(np.median(positive)) if positive.size else 0.0, eps)

        if positive.size:
            local_q75[i] = max(float(np.percentile(positive, 75.0)), local_median[i])

    local_scale = np.maximum(
        0.5 * local_median + 0.5 * global_scale,
        eps,
    )

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    valid_float = valid.astype(float)
    zero_edge = valid & (d == 0.0)

    relative_distance = np.divide(
        d,
        endpoint_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    robust_global_distance = np.divide(
        d - global_low,
        global_spread,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    robust_global_distance = np.maximum(robust_global_distance, 0.0)

    local_evidence = np.exp(-np.minimum(relative_distance, 40.0))
    global_evidence = np.exp(-np.minimum(robust_global_distance, 40.0))

    endpoint_competition = np.zeros_like(d, dtype=float)
    for i in range(n):
        threshold = local_q75[i]
        endpoint_competition[i] = np.where(
            valid[i],
            np.exp(-np.minimum(d[i] / max(threshold, eps), 40.0)),
            0.0,
        )

    mutual_local_evidence = np.sqrt(
        np.maximum(endpoint_competition, 0.0)
        * np.maximum(endpoint_competition.T, 0.0)
    )

    ranks = np.full((n, n), n, dtype=int)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="stable")]
            ranks[i, order] = np.arange(order.size)

    rank_decay = np.exp(-ranks.astype(float) / 2.5)
    rank_decay[ranks >= n] = 0.0

    reciprocal_rank = np.sqrt(rank_decay * rank_decay.T)

    mutual_nearest = (
        valid
        & (ranks == 0)
        & (ranks.T == 0)
    ).astype(float)

    local_sum = local_scale[:, None] + local_scale[None, :]
    savings = np.divide(
        np.maximum(local_sum - d, 0.0),
        np.maximum(local_sum, eps),
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )

    endpoint_extreme = (
        valid
        & (d > 1.8 * local_q75[:, None])
        & (d > 1.8 * local_q75[None, :])
    )

    excess = np.maximum(
        relative_distance - 1.0,
        0.0,
    )
    long_penalty = np.ones_like(d, dtype=float)
    long_penalty[endpoint_extreme] = np.exp(
        -0.35 * np.minimum(excess[endpoint_extreme], 8.0)
    )

    score = (
        0.28 * local_evidence
        + 0.22 * global_evidence
        + 0.20 * mutual_local_evidence
        + 0.16 * reciprocal_rank
        + 0.09 * mutual_nearest
        + 0.05 * savings
    )

    score *= long_penalty
    score[zero_edge] = 1.0
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
