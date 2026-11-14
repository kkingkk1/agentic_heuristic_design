import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & off_diagonal
    )

    if not np.any(valid):
        return score

    values = distance_matrix[valid]
    positive = values[values > 0.0]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    row_affinity = np.zeros((n, n), dtype=float)
    col_affinity = np.zeros((n, n), dtype=float)
    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)
    row_entropy = np.zeros(n, dtype=float)
    continuation = np.zeros((n, n), dtype=float)
    destination_demand = np.zeros((n, n), dtype=float)

    row_best = np.full(n, global_scale, dtype=float)
    row_second = np.full(n, global_scale, dtype=float)
    col_best = np.full(n, global_scale, dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        row_values = distance_matrix[i, indices]
        order = indices[np.argsort(row_values, kind="mergesort")]
        ordered_values = distance_matrix[i, order]

        row_best[i] = float(ordered_values[0])
        if ordered_values.size > 1:
            row_second[i] = float(ordered_values[1])

        local_values = ordered_values[: min(4, ordered_values.size)]
        local_positive = local_values[local_values > 0.0]
        local_scale = (
            float(np.median(local_positive))
            if local_positive.size
            else global_scale
        )
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = global_scale

        row_affinity[i, indices] = np.exp(
            -row_values / (local_scale + eps)
        )

        count = max(1, indices.size - 1)
        for rank, j in enumerate(order):
            row_rank[i, j] = 1.0 - float(rank) / float(count)

        logits = -row_values / (global_scale + eps)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities) + eps
        entropy = -np.sum(
            probabilities * np.log(probabilities + eps)
        )
        row_entropy[i] = entropy / np.log(float(max(2, indices.size)))

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        col_values = distance_matrix[indices, j]
        order = indices[np.argsort(col_values, kind="mergesort")]
        ordered_values = distance_matrix[order, j]

        col_best[j] = float(ordered_values[0])

        local_values = ordered_values[: min(4, ordered_values.size)]
        local_positive = local_values[local_values > 0.0]
        local_scale = (
            float(np.median(local_positive))
            if local_positive.size
            else global_scale
        )
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = global_scale

        col_affinity[indices, j] = np.exp(
            -col_values / (local_scale + eps)
        )

        count = max(1, indices.size - 1)
        for rank, i in enumerate(order):
            col_rank[i, j] = 1.0 - float(rank) / float(count)

    row_max = np.maximum(
        np.max(row_affinity, axis=1, keepdims=True),
        eps,
    )
    col_max = np.maximum(
        np.max(col_affinity, axis=0, keepdims=True),
        eps,
    )

    row_soft = row_affinity / row_max
    col_soft = col_affinity / col_max

    local_quality = (
        0.55 * row_soft
        + 0.20 * col_soft
        + 0.25 * np.exp(
            -distance_matrix / (global_scale + eps)
        )
    )
    local_quality[~valid] = 0.0

    bidirectional_competition = np.sqrt(
        np.maximum(row_rank, 0.0)
        * np.maximum(col_rank, 0.0)
    )

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])
        if outgoing.size == 0:
            continue

        for j in outgoing:
            incoming = np.flatnonzero(valid[:, j])

            alternative_out = outgoing[outgoing != j]
            if alternative_out.size:
                alt_out = float(
                    np.min(distance_matrix[i, alternative_out])
                )
            else:
                alt_out = row_second[i]

            alternative_in = incoming[incoming != i]
            if alternative_in.size:
                alt_in = float(
                    np.min(distance_matrix[alternative_in, j])
                )
            else:
                alt_in = col_best[j]

            candidate = float(distance_matrix[i, j])

            out_regret = max(alt_out - candidate, 0.0) / (
                alt_out + candidate + eps
            )
            in_regret = max(alt_in - candidate, 0.0) / (
                alt_in + candidate + eps
            )

            # Destination demand: favor edges entering nodes that have
            # relatively few strong incoming alternatives.
            incoming_quality = col_soft[:, j]
            strong_incoming = np.sum(
                incoming_quality >= 0.65
            )
            demand = 1.0 / (
                1.0 + 0.35 * max(0.0, strong_incoming - 1.0)
            )
            destination_demand[i, j] = demand

            # Two-step forward lookahead with explicit immediate-cycle
            # avoidance. The second-best continuation protects against
            # brittle dependence on one downstream edge.
            future = [
                k for k in np.flatnonzero(valid[j])
                if k != i and k != j
            ]

            if future:
                future_values = distance_matrix[j, future]
                future_order = np.argsort(
                    future_values,
                    kind="mergesort",
                )
                best_future = float(
                    future_values[future_order[0]]
                )
                if len(future_order) > 1:
                    second_future = float(
                        future_values[future_order[1]]
                    )
                else:
                    second_future = best_future

                future_affinity = np.exp(
                    -future_values / (global_scale + eps)
                )
                future_mass = float(
                    np.sum(future_affinity)
                )
                future_peak = float(
                    np.max(future_affinity)
                )
                downstream_diversity = future_mass / (
                    future_peak + eps
                )
                downstream_diversity = min(
                    downstream_diversity / float(max(2, len(future))),
                    1.0,
                )
            else:
                best_future = global_scale
                second_future = global_scale
                downstream_diversity = 0.0

            delayed_cost = (
                candidate
                + 0.52 * best_future
                + 0.18 * second_future
                + 0.16 * row_best[j]
                + 0.14 * row_second[j]
            )

            potential = 1.0 / (
                1.0 + delayed_cost / (global_scale + eps)
            )

            potential *= (
                0.72
                + 0.18 * row_entropy[j]
                + 0.10 * downstream_diversity
            )

            regret = 0.5 * (out_regret + in_regret)

            continuation[i, j] = (
                0.72 * potential
                + 0.18 * regret
                + 0.10 * destination_demand[i, j]
            )

    continuation_max = np.maximum(
        np.max(continuation, axis=1, keepdims=True),
        eps,
    )
    continuation_normalized = continuation / continuation_max

    # Penalize edges that are nearly interchangeable with many edges
    # sharing the same source or destination. This encourages global
    # coverage while retaining high-quality alternatives.
    redundancy = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size <= 1:
            continue

        row_values = distance_matrix[i, indices]
        row_scale = np.median(
            row_values[row_values > 0.0]
        ) if np.any(row_values > 0.0) else global_scale

        similarities = np.exp(
            -np.abs(
                row_values[:, None] - row_values[None, :]
            ) / (row_scale + eps)
        )
        np.fill_diagonal(similarities, 0.0)
        redundancy[i, indices] += np.mean(
            similarities,
            axis=1,
        )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size <= 1:
            continue

        col_values = distance_matrix[indices, j]
        col_scale = np.median(
            col_values[col_values > 0.0]
        ) if np.any(col_values > 0.0) else global_scale

        similarities = np.exp(
            -np.abs(
                col_values[:, None] - col_values[None, :]
            ) / (col_scale + eps)
        )
        np.fill_diagonal(similarities, 0.0)
        redundancy[indices, j] += 0.5 * np.mean(
            similarities,
            axis=1,
        )

    redundancy = np.clip(redundancy, 0.0, 1.5)
    diversity_bonus = 1.0 - 0.28 * redundancy

    score = (
        0.29 * local_quality
        + 0.22 * bidirectional_competition
        + 0.22 * continuation_normalized
        + 0.13 * destination_demand
        + 0.09 * row_rank
        + 0.05 * diversity_bonus
    )

    score *= diversity_bonus
    score[~valid] = 0.0
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    return score
