import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, symmetric and mildly globally calibrated TSP priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    both_valid = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = finite.T & ~finite

    cost = np.full((n, n), np.inf, dtype=float)
    cost[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    cost[only_forward] = d[only_forward]
    cost[only_backward] = d.T[only_backward]
    np.fill_diagonal(cost, np.inf)

    valid = np.isfinite(cost)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    positive = cost[valid & (cost > 0.0)]

    if positive.size:
        q10, q25, q50, q75 = np.percentile(
            positive, [10.0, 25.0, 50.0, 75.0]
        )
        global_scale = max(0.7 * float(q50) + 0.3 * float(q25), eps)
        global_spread = max(float(q75 - q25), 0.25 * global_scale, eps)
    else:
        q10 = q50 = 0.0
        global_scale = 1.0
        global_spread = 1.0

    neighborhood = min(6, max(3, n - 1))
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = np.sort(cost[i, np.isfinite(cost[i])])
        if values.size == 0:
            continue

        nearest = values[:min(neighborhood, values.size)]
        positive_nearest = nearest[nearest > 0.0]

        if positive_nearest.size:
            local_mean = float(np.mean(positive_nearest))
            local_median = float(np.median(positive_nearest))
            local_scale[i] = max(
                0.65 * local_mean + 0.35 * local_median,
                eps,
            )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(np.outer(local_scale, local_scale), eps)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        local_ratio = np.clip(cost / pair_scale, 0.0, 12.0)
        local_shortness = np.exp(-local_ratio)

        # Low-weight global calibration adds exploration without
        # overwhelming locally reliable distance information.
        global_shortness = np.exp(
            -np.clip((cost - q10) / global_spread, -5.0, 20.0)
        )

    local_shortness[~valid] = 0.0
    global_shortness[~valid] = 0.0

    masked = np.where(valid, cost, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")

    ranks = np.full((n, n), float(n), dtype=float)
    row_ids = np.arange(n)[:, None]
    positions = np.arange(n, dtype=float)[None, :]
    ordered_values = masked[row_ids, order]
    ordered_valid = np.isfinite(ordered_values)

    ranks[row_ids, order] = np.where(
        ordered_valid,
        positions,
        float(n),
    )

    rank_cap = max(float(neighborhood), 1.0)
    endpoint_quality = np.clip(
        1.0 - ranks / rank_cap,
        0.0,
        1.0,
    )
    rank_consensus = np.sqrt(
        np.maximum(endpoint_quality * endpoint_quality.T, 0.0)
    )

    reciprocal = (
        (ranks < float(neighborhood))
        & (ranks.T < float(neighborhood))
    ).astype(float)

    # Reserve only the two strongest local candidates per endpoint.
    reserve = (ranks < 2.0).astype(float)
    reserve_both = reserve * reserve.T
    reserve_one = np.clip(reserve + reserve.T, 0.0, 1.0)
    reserve_factor = (
        1.0
        + 0.08 * reserve_both
        + 0.025 * (reserve_one - reserve_both)
    )

    endpoint_margin = np.maximum(
        0.0,
        1.0 - ranks / float(max(n - 1, 1)),
    )
    mutual_margin = np.sqrt(
        np.maximum(endpoint_margin * endpoint_margin.T, 0.0)
    )

    base = (
        0.80 * local_shortness
        + 0.20 * global_shortness
    )

    scores = (
        base
        * (0.84 + 0.16 * rank_consensus)
        * (1.0 + 0.08 * reciprocal)
        * (0.97 + 0.03 * mutual_margin)
        * reserve_factor
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
