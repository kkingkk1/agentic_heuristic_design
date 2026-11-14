import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & off_diagonal
    )

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    valid_values = distance_matrix[valid]
    positive_values = valid_values[valid_values > 0.0]

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
    broad_affinity = np.zeros((n, n), dtype=float)
    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)

    best_out = np.full(n, np.inf, dtype=float)
    second_out = np.full(n, np.inf, dtype=float)
    best_in = np.full(n, np.inf, dtype=float)
    second_in = np.full(n, np.inf, dtype=float)

    out_counts = np.sum(valid, axis=1).astype(float)

    for i in range(n):
        destinations = np.flatnonzero(valid[i])
        if destinations.size == 0:
            continue

        values = distance_matrix[i, destinations]
        positive_local = values[values > 0.0]

        if positive_local.size:
            local_scale = float(np.median(positive_local))
        else:
            local_scale = global_scale

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = global_scale

        broad_scale = float(np.percentile(values, 75.0))
        if not np.isfinite(broad_scale) or broad_scale <= 0.0:
            broad_scale = global_scale

        local_affinity[i, destinations] = np.exp(
            -values / (local_scale + eps)
        )
        broad_affinity[i, destinations] = np.exp(
            -values / (broad_scale + eps)
        )

        order = destinations[np.argsort(values, kind="stable")]
        outgoing_rank[i, order] = 1.0 / (
            1.0 + np.arange(order.size, dtype=float)
        )

        ordered_values = np.sort(values)
        best_out[i] = ordered_values[0]
        if ordered_values.size > 1:
            second_out[i] = ordered_values[1]

        for j in destinations:
            alternatives = destinations[destinations != j]
            edge = float(distance_matrix[i, j])

            if alternatives.size:
                best_alternative = float(
                    np.min(distance_matrix[i, alternatives])
                )
                source_gain = max(best_alternative - edge, 0.0) / (
                    best_alternative + edge + eps
                )
            else:
                source_gain = 0.0

            incoming = np.flatnonzero(valid[:, j])
            alternatives_in = incoming[incoming != i]

            if alternatives_in.size:
                best_incoming = float(
                    np.min(distance_matrix[alternatives_in, j])
                )
                target_gain = max(best_incoming - edge, 0.0) / (
                    best_incoming + edge + eps
                )
            else:
                target_gain = 0.0

            regret[i, j] = 0.5 * (source_gain + target_gain)

    for j in range(n):
        sources = np.flatnonzero(valid[:, j])
        if sources.size == 0:
            continue

        values = distance_matrix[sources, j]
        order = sources[np.argsort(values, kind="stable")]
        incoming_rank[order, j] = 1.0 / (
            1.0 + np.arange(order.size, dtype=float)
        )

        ordered_values = np.sort(values)
        best_in[j] = ordered_values[0]
        if ordered_values.size > 1:
            second_in[j] = ordered_values[1]

    bidirectional_rank = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    continuation = np.zeros((n, n), dtype=float)

    for i in range(n):
        destinations = np.flatnonzero(valid[i])

        for j in destinations:
            future_out = (
                best_out[j]
                if np.isfinite(best_out[j])
                else global_scale
            )
            future_out_2 = (
                second_out[j]
                if np.isfinite(second_out[j])
                else future_out
            )
            future_in = (
                best_in[i]
                if np.isfinite(best_in[i])
                else global_scale
            )
            future_in_2 = (
                second_in[i]
                if np.isfinite(second_in[i])
                else future_in
            )

            delayed_cost = (
                float(distance_matrix[i, j])
                + 0.50 * future_out
                + 0.20 * future_out_2
                + 0.20 * future_in
                + 0.10 * future_in_2
            )

            continuation[i, j] = 1.0 / (
                1.0 + delayed_cost / (global_scale + eps)
            )

    continuation_max = np.maximum(
        np.max(continuation, axis=1, keepdims=True),
        eps,
    )
    continuation_normalized = continuation / continuation_max

    for i in range(n):
        row_valid = valid[i]
        degree = out_counts[i]

        if degree <= 2:
            local_weight = 0.31
            rank_weight = 0.36
            regret_weight = 0.29
            exploration_weight = 0.04
        elif degree <= 5:
            local_weight = 0.34
            rank_weight = 0.31
            regret_weight = 0.24
            exploration_weight = 0.11
        else:
            local_weight = 0.36
            rank_weight = 0.28
            regret_weight = 0.21
            exploration_weight = 0.15

        score = (
            local_weight * local_affinity[i, row_valid]
            + rank_weight * bidirectional_rank[i, row_valid]
            + regret_weight * regret[i, row_valid]
            + exploration_weight
            * (
                0.55 * continuation_normalized[i, row_valid]
                + 0.25 * broad_affinity[i, row_valid]
                + 0.20 * global_affinity[i, row_valid]
            )
        )

        result[i, row_valid] = score

    result[~valid] = 0.0

    for i in range(n):
        maximum = float(np.max(result[i]))
        if maximum > 0.0 and np.isfinite(maximum):
            result[i] /= maximum

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    result = np.maximum(result, 0.0)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return result
