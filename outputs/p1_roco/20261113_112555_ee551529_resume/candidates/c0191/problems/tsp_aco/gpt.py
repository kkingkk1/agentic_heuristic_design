import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative but globally diversified TSP edge priors."""
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
    forward_only = finite & ~finite.T
    backward_only = finite.T & ~finite

    cost = np.full((n, n), np.inf, dtype=float)
    cost[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    cost[forward_only] = d[forward_only]
    cost[backward_only] = d.T[backward_only]
    np.fill_diagonal(cost, np.inf)

    valid = np.isfinite(cost)
    if not np.any(valid):
        return scores

    positive = cost[valid & (cost > 0.0)]
    eps = np.finfo(float).eps

    if positive.size:
        q25, median, q75 = np.percentile(positive, [25.0, 50.0, 75.0])
    else:
        q25 = median = q75 = 1.0

    global_scale = max(float(median), eps)
    width = max(float(q75 - q25), 0.25 * global_scale, eps)
    neighborhood = min(6, max(3, n - 1))

    local_scale = np.full(n, global_scale, dtype=float)
    continuation = np.full(n, global_scale, dtype=float)
    ranks = np.full((n, n), float(n), dtype=float)
    entropy = np.zeros(n, dtype=float)

    for i in range(n):
        row = cost[i]
        values = np.sort(row[np.isfinite(row)])

        if values.size == 0:
            continue

        k = min(neighborhood, values.size)
        local_values = values[:k]

        if local_values.size >= 4:
            local_values = local_values[:-1]

        positive_local = local_values[local_values > 0.0]
        if positive_local.size:
            local_scale[i] = max(float(np.mean(positive_local)), eps)

        if values.size > 1:
            alternatives = values[1:min(values.size, k + 1)]
            positive_alternatives = alternatives[alternatives > 0.0]
            if positive_alternatives.size:
                continuation[i] = max(
                    float(np.mean(positive_alternatives)),
                    eps,
                )
            else:
                continuation[i] = local_scale[i]
        else:
            continuation[i] = local_scale[i]

        order = np.argsort(row, kind="stable")
        for position, j in enumerate(order):
            if np.isfinite(row[j]):
                ranks[i, j] = float(position)

        finite_row = np.isfinite(row)
        logits = -np.clip(row[finite_row] / local_scale[i], 0.0, 40.0)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= max(float(np.sum(probabilities)), eps)

        if probabilities.size > 1:
            entropy[i] = -float(
                np.sum(probabilities * np.log(np.maximum(probabilities, eps)))
            ) / np.log(probabilities.size)

    local_scale = np.maximum(local_scale, eps)
    continuation = np.maximum(continuation, eps)

    pair_scale = np.sqrt(np.outer(local_scale, local_scale))
    relative_distance = cost / np.maximum(pair_scale, eps)

    shortness = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    rank_affinity = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(rank_affinity * rank_affinity.T)

    reciprocal = (
        (ranks < float(neighborhood))
        & (ranks.T < float(neighborhood))
    ).astype(float)

    competitive = np.clip(
        1.0 - ranks / max(float(neighborhood), 1.0),
        0.0,
        1.0,
    )
    future_potential = np.sqrt(competitive * competitive.T)

    global_shortness = np.exp(
        -np.clip((cost - q25) / width, -8.0, 30.0)
    )

    endpoint_entropy = np.sqrt(np.outer(entropy, entropy))
    diversity_bonus = 0.96 + 0.04 * endpoint_entropy

    edge_cost = np.maximum(cost, eps)
    alternative_ratio = np.sqrt(
        np.outer(continuation, continuation)
    ) / edge_cost
    alternative_quality = np.exp(
        -np.clip(alternative_ratio / 3.0, 0.0, 12.0)
    )
    future_bonus = 0.98 + 0.02 * alternative_quality

    scores = (
        shortness
        * (0.78 + 0.22 * rank_consensus)
        * (1.0 + 0.14 * reciprocal)
        * (0.94 + 0.06 * future_potential)
        * (0.96 + 0.04 * global_shortness)
        * diversity_bonus
        * future_bonus
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
