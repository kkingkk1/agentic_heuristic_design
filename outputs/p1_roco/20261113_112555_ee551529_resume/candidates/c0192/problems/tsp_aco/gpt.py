import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse and long-term-aware TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    if positive.size:
        q20, q25, q50, q75, q80 = np.percentile(
            positive, [20.0, 25.0, 50.0, 75.0, 80.0]
        )
    else:
        q20 = q25 = q50 = q75 = q80 = 1.0

    eps = np.finfo(float).eps
    global_scale = max(float(q50), eps)
    spread = max(float(q75 - q25), 0.25 * global_scale, eps)

    reverse_finite = finite.T
    both_valid = finite & reverse_finite
    forward_only = finite & ~reverse_finite
    backward_only = reverse_finite & ~finite

    cost = np.full((n, n), np.inf, dtype=float)
    cost[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    cost[forward_only] = d[forward_only]
    cost[backward_only] = d.T[backward_only]
    np.fill_diagonal(cost, np.inf)

    valid = np.isfinite(cost)
    if not np.any(valid):
        return scores

    k = min(max(3, int(np.ceil(np.sqrt(max(n, 2))))), 7, n - 1)

    local_scale = np.full(n, global_scale, dtype=float)
    continuation = np.full(n, global_scale, dtype=float)
    entropy = np.zeros(n, dtype=float)
    ranks = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = cost[i].copy()
        values = np.sort(row[np.isfinite(row)])

        if values.size == 0:
            continue

        neighborhood = values[:k]
        if neighborhood.size >= 4:
            neighborhood = neighborhood[:-1]

        local_scale[i] = max(float(np.mean(neighborhood)), eps)

        if values.size > 1:
            alternatives = values[1:min(values.size, k + 1)]
            continuation[i] = max(float(np.mean(alternatives)), eps)
        else:
            continuation[i] = local_scale[i]

        usable = np.isfinite(row)
        logits = -np.clip(row[usable] / local_scale[i], 0.0, 35.0)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= max(float(np.sum(probabilities)), eps)

        if probabilities.size > 1:
            raw_entropy = -np.sum(
                probabilities * np.log(np.maximum(probabilities, eps))
            )
            entropy[i] = float(raw_entropy / np.log(probabilities.size))

        order = np.argsort(
            np.where(np.isfinite(row), row, np.inf),
            kind="stable",
        )
        for rank, j in enumerate(order[:k]):
            if np.isfinite(row[j]):
                ranks[i, j] = np.exp(-rank / max(k - 1, 1))

    local_scale = np.maximum(local_scale, eps)
    continuation = np.maximum(continuation, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        local_ratio = cost / np.sqrt(
            np.outer(local_scale, local_scale)
        )
        global_ratio = cost / global_scale

        local_shortness = np.exp(-np.clip(local_ratio, 0.0, 30.0))

        global_shortness = np.exp(
            -np.clip((cost - q20) / spread, -6.0, 30.0)
        )

        quantile_gate = 0.90 + 0.10 * np.exp(
            -np.clip((cost - q50) / max(spread, eps), 0.0, 12.0)
        )

        rank_consensus = np.sqrt(
            np.maximum(ranks * ranks.T, 0.0)
        )

        endpoint_entropy = np.sqrt(
            np.outer(entropy, entropy)
        )

        # Reward moderately diverse nodes, but avoid rewarding maximally
        # diffuse and uniformly poor neighborhoods.
        entropy_focus = 4.0 * endpoint_entropy * (1.0 - endpoint_entropy)
        diversity_bonus = 0.95 + 0.10 * entropy_focus

        edge_cost = np.maximum(cost, eps)
        alternative_scale = np.sqrt(
            np.outer(continuation, continuation)
        )

        opportunity_ratio = alternative_scale / edge_cost

        # A ratio above one means the current edge is unusually important.
        # Penalize only clear lock-in cases, and protect very short edges.
        log_opportunity = np.log1p(np.maximum(opportunity_ratio, 0.0))
        lock_in = 1.0 / (
            1.0 + np.exp(-2.2 * (log_opportunity - np.log(2.0)))
        )

        short_edge_protection = np.exp(
            -np.clip(1.5 * local_ratio, 0.0, 12.0)
        )

        long_term_penalty = 1.0 - 0.16 * lock_in * (
            1.0 - short_edge_protection
        )

        # Balance opportunity costs at the two endpoints. Highly asymmetric
        # dependence is mildly discouraged, without rejecting useful bridges.
        endpoint_balance = np.minimum(
            opportunity_ratio,
            1.0 / np.maximum(opportunity_ratio, eps),
        )
        endpoint_balance = np.clip(endpoint_balance, 0.0, 1.0)
        balance_bonus = 0.94 + 0.08 * np.sqrt(endpoint_balance)

        scale_agreement = np.exp(
            -np.clip(
                np.abs(
                    np.log1p(np.maximum(local_ratio, 0.0))
                    - np.log1p(np.maximum(global_ratio, 0.0))
                ),
                0.0,
                8.0,
            )
        )
        global_calibration = 0.90 + 0.10 * scale_agreement

        scores = (
            (0.56 * local_shortness + 0.44 * global_shortness)
            * (0.70 + 0.30 * rank_consensus)
            * diversity_bonus
            * long_term_penalty
            * balance_bonus
            * global_calibration
            * quantile_gate
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
