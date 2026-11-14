import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be a square matrix")

    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & (~diagonal)
    )

    if not np.any(valid):
        return result

    positive = distance_matrix[valid]
    positive = positive[positive > 0.0]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)
    local_affinity = np.zeros((n, n), dtype=float)
    reverse_affinity = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)
    destination_demand = np.zeros((n, n), dtype=float)
    continuation = np.zeros((n, n), dtype=float)

    out_counts = valid.sum(axis=1)

    for i in range(n):
        destinations = np.flatnonzero(valid[i])
        if destinations.size == 0:
            continue

        values = distance_matrix[i, destinations]
        positive_local = values[values > 0.0]
        local_scale = (
            float(np.median(positive_local))
            if positive_local.size
            else global_scale
        )
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = global_scale

        local_affinity[i, destinations] = np.exp(
            -values / (local_scale + eps)
        )

        order = destinations[np.argsort(values, kind="stable")]
        count = max(1, destinations.size - 1)
        for rank, j in enumerate(order):
            outgoing_rank[i, j] = 1.0 - float(rank) / float(count)

        reverse_values = distance_matrix[destinations, i]
        reverse_valid = np.isfinite(reverse_values) & (reverse_values >= 0.0)
        reverse_affinity[i, destinations[reverse_valid]] = np.exp(
            -reverse_values[reverse_valid] / (local_scale + eps)
        )

        finite_values = values[np.isfinite(values)]
        if finite_values.size > 1:
            spread = float(np.std(finite_values))
            contrast = spread / (float(np.mean(finite_values)) + eps)
            contrast = float(np.clip(contrast, 0.0, 1.0))
        else:
            contrast = 0.0

        if destinations.size <= 2:
            local_weight = 0.24 + 0.14 * contrast
            rank_weight = 0.34 - 0.08 * contrast
            regret_weight = 0.35
            reverse_weight = 0.05
        else:
            local_weight = 0.34 + 0.18 * contrast
            rank_weight = 0.31 - 0.11 * contrast
            regret_weight = 0.29
            reverse_weight = 0.06

        for j in destinations:
            edge = float(distance_matrix[i, j])

            alternatives = destinations[destinations != j]
            if alternatives.size:
                alternative_values = distance_matrix[i, alternatives]
                ordered = np.sort(alternative_values, kind="stable")
                best = float(ordered[0])
                second = float(ordered[1]) if ordered.size > 1 else best

                source_regret = (
                    0.70 * max(best - edge, 0.0)
                    + 0.30 * max(second - best, 0.0)
                ) / (best + edge + eps)
            else:
                source_regret = 0.0

            incoming = np.flatnonzero(valid[:, j])
            alternatives_in = incoming[incoming != i]
            if alternatives_in.size:
                incoming_values = distance_matrix[alternatives_in, j]
                ordered = np.sort(incoming_values, kind="stable")
                best = float(ordered[0])
                second = float(ordered[1]) if ordered.size > 1 else best

                target_regret = (
                    0.70 * max(best - edge, 0.0)
                    + 0.30 * max(second - best, 0.0)
                ) / (best + edge + eps)
            else:
                target_regret = 0.0

            regret[i, j] = 0.5 * (
                source_regret + target_regret
            )

            incoming_quality = np.zeros(incoming.size, dtype=float)
            for index, source in enumerate(incoming):
                incoming_edge = float(distance_matrix[source, j])
                incoming_scale = global_scale
                incoming_positive = distance_matrix[
                    valid[:, j], j
                ]
                incoming_positive = incoming_positive[
                    incoming_positive > 0.0
                ]
                if incoming_positive.size:
                    incoming_scale = float(
                        np.median(incoming_positive)
                    )
                incoming_quality[index] = np.exp(
                    -incoming_edge / (incoming_scale + eps)
                )

            strong_incoming = np.sum(incoming_quality >= 0.65)
            destination_demand[i, j] = 1.0 / (
                1.0 + 0.35 * max(float(strong_incoming) - 1.0, 0.0)
            )

            future = np.flatnonzero(valid[j])
            future = future[(future != i) & (future != j)]
            if future.size:
                future_values = distance_matrix[j, future]
                ordered_future = np.sort(
                    future_values,
                    kind="stable",
                )
                best_future = float(ordered_future[0])
                second_future = (
                    float(ordered_future[1])
                    if ordered_future.size > 1
                    else best_future
                )
            else:
                best_future = global_scale
                second_future = global_scale

            delayed_cost = (
                edge
                + 0.58 * best_future
                + 0.18 * second_future
            )
            continuation[i, j] = 1.0 / (
                1.0 + delayed_cost / (global_scale + eps)
            )

    bidirectional_rank = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    for j in range(n):
        sources = np.flatnonzero(valid[:, j])
        if sources.size == 0:
            continue

        values = distance_matrix[sources, j]
        order = sources[np.argsort(values, kind="stable")]
        count = max(1, sources.size - 1)
        for rank, i in enumerate(order):
            incoming_rank[i, j] = 1.0 - float(rank) / float(count)

    bidirectional_rank = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    for i in range(n):
        destinations = np.flatnonzero(valid[i])
        if destinations.size == 0:
            continue

        continuation_values = continuation[i, destinations]
        maximum = float(np.max(continuation_values))
        if maximum > 0.0 and np.isfinite(maximum):
            continuation_values = continuation_values / maximum

        score = (
            local_weight * local_affinity[i, destinations]
            + rank_weight * bidirectional_rank[i, destinations]
            + regret_weight * regret[i, destinations]
            + reverse_weight * reverse_affinity[i, destinations]
            + 0.08 * destination_demand[i, destinations]
            + 0.10 * continuation_values
        )

        result[i, destinations] = np.maximum(score, 0.0)

        maximum = float(np.max(result[i]))
        if maximum > 0.0 and np.isfinite(maximum):
            result[i] /= maximum

    result[~valid] = 0.0
    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    result = np.maximum(result, 0.0)
    np.fill_diagonal(result, 0.0)
    return result
