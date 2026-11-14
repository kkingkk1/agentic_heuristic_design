import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
    ):
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
    positive_values = values[values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -distance_matrix[valid] / (global_scale + eps)
    )

    local_affinity = np.zeros((n, n), dtype=float)
    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)
    continuation = np.zeros((n, n), dtype=float)
    destination_demand = np.zeros((n, n), dtype=float)
    redundancy = np.zeros((n, n), dtype=float)

    out_counts = np.sum(valid, axis=1).astype(float)

    row_best = np.full(n, global_scale, dtype=float)
    row_second = np.full(n, global_scale, dtype=float)
    col_best = np.full(n, global_scale, dtype=float)

    for i in range(n):
        destinations = np.flatnonzero(valid[i])
        if destinations.size == 0:
            continue

        row_values = distance_matrix[i, destinations]
        positive_local = row_values[row_values > 0.0]
        local_scale = (
            float(np.median(positive_local))
            if positive_local.size
            else global_scale
        )

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = global_scale

        local_affinity[i, destinations] = np.exp(
            -row_values / (local_scale + eps)
        )

        order = destinations[np.argsort(row_values, kind="stable")]
        ordered_values = distance_matrix[i, order]
        row_best[i] = float(ordered_values[0])

        if ordered_values.size > 1:
            row_second[i] = float(ordered_values[1])

        for rank, j in enumerate(order):
            outgoing_rank[i, j] = 1.0 / (1.0 + float(rank))

        if destinations.size > 1:
            similarities = np.exp(
                -np.abs(row_values[:, None] - row_values[None, :])
                / (local_scale + eps)
            )
            np.fill_diagonal(similarities, 0.0)
            redundancy[i, destinations] += np.mean(
                similarities,
                axis=1,
            )

    for j in range(n):
        sources = np.flatnonzero(valid[:, j])
        if sources.size == 0:
            continue

        column_values = distance_matrix[sources, j]
        positive_local = column_values[column_values > 0.0]
        local_scale = (
            float(np.median(positive_local))
            if positive_local.size
            else global_scale
        )

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = global_scale

        order = sources[np.argsort(column_values, kind="stable")]
        col_best[j] = float(distance_matrix[order[0], j])

        for rank, i in enumerate(order):
            incoming_rank[i, j] = 1.0 / (1.0 + float(rank))

        if sources.size > 1:
            similarities = np.exp(
                -np.abs(
                    column_values[:, None] - column_values[None, :]
                )
                / (local_scale + eps)
            )
            np.fill_diagonal(similarities, 0.0)
            redundancy[sources, j] += 0.5 * np.mean(
                similarities,
                axis=1,
            )

    bidirectional_rank = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    for i in range(n):
        destinations = np.flatnonzero(valid[i])
        if destinations.size == 0:
            continue

        for j in destinations:
            edge = float(distance_matrix[i, j])

            alternatives = destinations[destinations != j]
            if alternatives.size:
                best_alternative = float(
                    np.min(distance_matrix[i, alternatives])
                )
            else:
                best_alternative = row_second[i]

            incoming = np.flatnonzero(valid[:, j])
            alternatives_in = incoming[incoming != i]
            if alternatives_in.size:
                best_incoming = float(
                    np.min(distance_matrix[alternatives_in, j])
                )
            else:
                best_incoming = col_best[j]

            source_gain = max(best_alternative - edge, 0.0) / (
                best_alternative + edge + eps
            )
            target_gain = max(best_incoming - edge, 0.0) / (
                best_incoming + edge + eps
            )
            regret[i, j] = 0.5 * (source_gain + target_gain)

            incoming_quality = np.zeros(incoming.size, dtype=float)
            if incoming.size:
                incoming_quality = np.exp(
                    -distance_matrix[incoming, j]
                    / (global_scale + eps)
                )
                incoming_quality /= max(
                    float(np.max(incoming_quality)),
                    eps,
                )

            strong_incoming = float(
                np.sum(incoming_quality >= 0.65)
            )
            destination_demand[i, j] = 1.0 / (
                1.0 + 0.35 * max(strong_incoming - 1.0, 0.0)
            )

            future = np.flatnonzero(valid[j])
            future = future[(future != i) & (future != j)]

            if future.size:
                future_values = distance_matrix[j, future]
                order_future = np.argsort(
                    future_values,
                    kind="stable",
                )
                ordered_future = future_values[order_future]

                best_future = float(ordered_future[0])
                second_future = (
                    float(ordered_future[1])
                    if ordered_future.size > 1
                    else best_future
                )

                future_affinity = np.exp(
                    -future_values / (global_scale + eps)
                )
                future_peak = float(np.max(future_affinity))
                future_mass = float(np.sum(future_affinity))
                diversity = future_mass / (future_peak + eps)
                diversity /= float(max(2, future.size))
                diversity = min(max(diversity, 0.0), 1.0)
            else:
                best_future = global_scale
                second_future = global_scale
                diversity = 0.0

            delayed_cost = (
                edge
                + 0.45 * best_future
                + 0.15 * second_future
                + 0.12 * row_best[j]
                + 0.10 * row_second[j]
            )

            potential = 1.0 / (
                1.0 + delayed_cost / (global_scale + eps)
            )
            potential *= 0.85 + 0.15 * diversity

            continuation[i, j] = (
                0.72 * potential
                + 0.18 * regret[i, j]
                + 0.10 * destination_demand[i, j]
            )

    continuation_max = np.maximum(
        np.max(continuation, axis=1, keepdims=True),
        eps,
    )
    continuation_normalized = continuation / continuation_max

    redundancy = np.clip(redundancy, 0.0, 1.5)
    diversity_factor = 1.0 - 0.20 * redundancy

    for i in range(n):
        row_valid = valid[i]
        degree = out_counts[i]

        if degree <= 2:
            local_weight = 0.30
            rank_weight = 0.35
            regret_weight = 0.27
            continuation_weight = 0.05
            demand_weight = 0.03
        else:
            local_weight = 0.36
            rank_weight = 0.30
            regret_weight = 0.23
            continuation_weight = 0.07
            demand_weight = 0.04

        score[i, row_valid] = (
            local_weight * local_affinity[i, row_valid]
            + rank_weight * bidirectional_rank[i, row_valid]
            + regret_weight * regret[i, row_valid]
            + continuation_weight
            * continuation_normalized[i, row_valid]
            + demand_weight * destination_demand[i, row_valid]
        )

    score *= diversity_factor
    score[~valid] = 0.0
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    score = np.maximum(score, 0.0)

    for i in range(n):
        maximum = float(np.max(score[i]))
        if maximum > 0.0 and np.isfinite(maximum):
            score[i] /= maximum

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score
