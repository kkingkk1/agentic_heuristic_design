import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust local TSP edge priors with lightweight global correction."""
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
        q20, q25, q50, q75 = np.percentile(
            positive, [20.0, 25.0, 50.0, 75.0]
        )
        global_scale = 0.7 * float(q50) + 0.3 * float(q25)
    else:
        q20 = q25 = q50 = q75 = 1.0
        global_scale = 1.0

    global_scale = max(global_scale, eps)
    spread = max(float(q75 - q25), 0.25 * global_scale, eps)

    neighborhood = min(6, max(3, n - 1))
    local_scale = np.full(n, global_scale, dtype=float)
    continuation = np.full(n, global_scale, dtype=float)
    entropy = np.zeros(n, dtype=float)

    ranks = np.full((n, n), float(n), dtype=float)

    for i in range(n):
        row = cost[i]
        values = np.sort(row[np.isfinite(row)])

        if values.size == 0:
            continue

        k = min(neighborhood, values.size)
        nearest = values[:k]
        positive_nearest = nearest[nearest > 0.0]

        if positive_nearest.size:
            local_mean = float(np.mean(positive_nearest))
            local_median = float(np.median(positive_nearest))
            local_scale[i] = 0.65 * local_mean + 0.35 * local_median

        if values.size > 1:
            alternatives = values[1:min(values.size, k + 1)]
            if alternatives.size:
                continuation[i] = max(float(np.mean(alternatives)), eps)
            else:
                continuation[i] = local_scale[i]
        else:
            continuation[i] = local_scale[i]

        usable = np.isfinite(row)
        logits = -np.clip(row[usable] / max(local_scale[i], eps), 0.0, 35.0)
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
        valid_order = order[:k]
        for rank, j in enumerate(valid_order):
            if np.isfinite(row[j]):
                ranks[i, j] = float(rank)

    local_scale = np.maximum(
        np.where(np.isfinite(local_scale), local_scale, global_scale),
        eps,
    )
    continuation = np.maximum(
        np.where(np.isfinite(continuation), continuation, global_scale),
        eps,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pair_local_scale = np.sqrt(
            np.maximum(np.outer(local_scale, local_scale), eps)
        )
        local_ratio = cost / pair_local_scale
        global_ratio = cost / global_scale

        local_shortness = np.exp(-np.clip(local_ratio, 0.0, 40.0))
        global_shortness = np.exp(-np.clip(global_ratio, 0.0, 40.0))

        # Exploiter-dominant shortness with a weak Explorer global correction.
        shortness = (
            0.78 * local_shortness
            + 0.14 * np.exp(-np.clip(cost / global_scale, 0.0, 40.0))
            + 0.08 * np.exp(
                -np.clip((cost - q20) / spread, -6.0, 30.0)
            )
        )

        rank_affinity = 1.0 / (1.0 + ranks)
        rank_consensus = np.sqrt(
            np.maximum(rank_affinity * rank_affinity.T, 0.0)
        )

        reciprocal = (
            (ranks < float(neighborhood))
            & (ranks.T < float(neighborhood))
        ).astype(float)

        rank_cap = max(float(neighborhood), 1.0)
        endpoint_competitiveness = np.clip(
            1.0 - ranks / rank_cap,
            0.0,
            1.0,
        )
        mutual_competitiveness = np.sqrt(
            endpoint_competitiveness
            * endpoint_competitiveness.T
        )

        endpoint_entropy = np.sqrt(
            np.maximum(np.outer(entropy, entropy), 0.0)
        )
        entropy_focus = 4.0 * endpoint_entropy * (1.0 - endpoint_entropy)
        diversity_correction = 0.985 + 0.03 * entropy_focus

        alternative_scale = np.sqrt(
            np.maximum(np.outer(continuation, continuation), eps)
        )
        opportunity_ratio = alternative_scale / np.maximum(cost, eps)

        log_opportunity = np.log1p(
            np.maximum(opportunity_ratio, 0.0)
        )
        lock_in = 1.0 / (
            1.0
            + np.exp(
                -2.0 * (log_opportunity - np.log(2.0))
            )
        )

        short_edge_protection = np.exp(
            -np.clip(1.7 * local_ratio, 0.0, 12.0)
        )
        long_term_penalty = 1.0 - 0.08 * lock_in * (
            1.0 - short_edge_protection
        )

        endpoint_balance = np.minimum(
            opportunity_ratio,
            1.0 / np.maximum(opportunity_ratio, eps),
        )
        balance_bonus = 0.98 + 0.03 * np.sqrt(
            np.clip(endpoint_balance, 0.0, 1.0)
        )

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
        global_calibration = 0.98 + 0.02 * scale_agreement

        scores = (
            shortness
            * (0.80 + 0.20 * rank_consensus)
            * (1.0 + 0.10 * reciprocal)
            * (0.96 + 0.04 * mutual_competitiveness)
            * diversity_correction
            * long_term_penalty
            * balance_bonus
            * global_calibration
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
