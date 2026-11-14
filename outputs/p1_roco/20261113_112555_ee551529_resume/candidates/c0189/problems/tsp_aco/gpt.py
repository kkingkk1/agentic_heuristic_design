import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse and long-term-aware TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    if positive.size:
        q25, q50, q75 = np.percentile(positive, [25.0, 50.0, 75.0])
    else:
        q25 = q50 = q75 = 1.0

    eps = np.finfo(float).eps
    scale = max(float(q50), eps)
    width = max(float(q75 - q25), 0.25 * scale, eps)

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
        return result

    k = min(max(4, int(np.ceil(np.sqrt(max(n, 2))))), 6, n - 1)

    local_scale = np.full(n, scale, dtype=float)
    continuation = np.full(n, scale, dtype=float)
    soft_entropy = np.zeros(n, dtype=float)
    local_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = cost[i].copy()
        row[~valid[i]] = np.inf
        values = np.sort(row[np.isfinite(row)])

        if values.size == 0:
            continue

        neighborhood = values[:k]

        # Truncated-neighborhood mean: remove the largest local value when
        # enough neighbors exist, reducing sensitivity to one bad edge.
        if neighborhood.size >= 4:
            trimmed = neighborhood[:-1]
        else:
            trimmed = neighborhood

        local_scale[i] = max(float(np.mean(trimmed)), eps)

        if values.size > 1:
            tail = values[1:min(values.size, k + 1)]
            continuation[i] = max(float(np.mean(tail)), eps)
        else:
            continuation[i] = local_scale[i]

        finite_row = np.isfinite(row)
        logits = -np.clip(row[finite_row] / local_scale[i], 0.0, 40.0)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= max(float(np.sum(probabilities)), eps)

        if probabilities.size > 1:
            entropy = -float(np.sum(probabilities * np.log(
                np.maximum(probabilities, eps)
            )))
            soft_entropy[i] = entropy / np.log(probabilities.size)

        order = np.argsort(row, kind="stable")
        for rank, j in enumerate(order[:k]):
            if np.isfinite(row[j]):
                local_rank[i, j] = np.exp(-rank / max(k - 1, 1))

    local_scale = np.maximum(local_scale, eps)
    continuation = np.maximum(continuation, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        local_relative = cost / np.sqrt(
            np.outer(local_scale, local_scale)
        )
        global_relative = cost / scale

        local_shortness = np.exp(-np.clip(local_relative, 0.0, 30.0))
        global_shortness = np.exp(
            -np.clip((cost - q25) / width, -8.0, 30.0)
        )

        rank_consensus = np.sqrt(
            np.maximum(local_rank * local_rank.T, 0.0)
        )

        reciprocal_bonus = 1.0 + 0.12 * rank_consensus

        # High-entropy endpoints retain several competitive continuations,
        # which supports global candidate coverage.
        endpoint_entropy = np.sqrt(
            np.outer(soft_entropy, soft_entropy)
        )
        diversity_bonus = 0.84 + 0.16 * endpoint_entropy

        edge_cost = np.maximum(cost, eps)
        alternative_ratio = np.sqrt(
            np.outer(continuation, continuation)
        ) / edge_cost

        # Alternatives should remain reasonably competitive. Very poor
        # alternatives indicate that the edge may create a brittle commitment.
        alternative_quality = np.exp(
            -np.clip(alternative_ratio / 3.0, 0.0, 12.0)
        )

        # Mildly discourage edges that are uniquely attractive relative to
        # both endpoints' alternatives, while preserving strong short edges.
        scarcity_penalty = 0.88 + 0.12 * np.exp(
            -np.clip(np.abs(np.log1p(alternative_ratio)), 0.0, 8.0)
        )

        future_potential = (
            0.72
            + 0.28 * alternative_quality
        ) * scarcity_penalty

        # Preserve edges that are simultaneously useful at local and global
        # scales; this avoids collapsing all mass onto one local regime.
        scale_agreement = np.exp(
            -np.clip(
                np.abs(np.log1p(local_relative) -
                       np.log1p(global_relative)),
                0.0,
                8.0,
            )
        )
        global_diversity = 0.82 + 0.18 * scale_agreement

        scores = (
            (0.60 * local_shortness + 0.40 * global_shortness)
            * (0.72 + 0.28 * rank_consensus)
            * reciprocal_bonus
            * future_potential
            * diversity_bonus
            * global_diversity
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
