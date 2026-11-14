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

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & (~diagonal)
    )

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps

    positive_values = distance_matrix[valid]
    positive_values = positive_values[positive_values > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)
    local_affinity = np.zeros((n, n), dtype=float)
    reverse_affinity = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)

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
        for rank, j in enumerate(order):
            outgoing_rank[i, j] = 1.0 / (1.0 + rank)

        for j in destinations:
            alternatives = destinations[destinations != j]
            edge = float(distance_matrix[i, j])

            if alternatives.size:
                alternative_values = distance_matrix[i, alternatives]
                alternative_values = alternative_values[
                    np.isfinite(alternative_values)
                    & (alternative_values >= 0.0)
                ]
                if alternative_values.size:
                    ordered = np.sort(alternative_values, kind="stable")
                    best = float(ordered[0])
                    second = float(ordered[1]) if ordered.size > 1 else best
                    indispensability = max(best - edge, 0.0)
                    separation = max(second - best, 0.0)
                    source_regret = (
                        0.70 * indispensability
                        + 0.30 * separation
                    ) / (best + edge + eps)
                else:
                    source_regret = 0.0
            else:
                source_regret = 0.0

            incoming = np.flatnonzero(valid[:, j])
            alternatives_in = incoming[incoming != i]

            if alternatives_in.size:
                incoming_values = distance_matrix[alternatives_in, j]
                incoming_values = incoming_values[
                    np.isfinite(incoming_values)
                    & (incoming_values >= 0.0)
                ]
                if incoming_values.size:
                    ordered = np.sort(incoming_values, kind="stable")
                    best = float(ordered[0])
                    second = float(ordered[1]) if ordered.size > 1 else best
                    indispensability = max(best - edge, 0.0)
                    separation = max(second - best, 0.0)
                    target_regret = (
                        0.70 * indispensability
                        + 0.30 * separation
                    ) / (best + edge + eps)
                else:
                    target_regret = 0.0
            else:
                target_regret = 0.0

            regret[i, j] = 0.5 * (source_regret + target_regret)

    for j in range(n):
        sources = np.flatnonzero(valid[:, j])
        if sources.size == 0:
            continue

        values = distance_matrix[sources, j]
        order = sources[np.argsort(values, kind="stable")]
        for rank, i in enumerate(order):
            incoming_rank[i, j] = 1.0 / (1.0 + rank)

    bidirectional_rank = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    for i in range(n):
        destinations = np.flatnonzero(valid[i])
        if destinations.size == 0:
            continue

        values = distance_matrix[i, destinations]
        local_scale = np.median(
            values[values > 0.0]
        ) if np.any(values > 0.0) else global_scale

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = global_scale

        reverse_values = distance_matrix[destinations, i]
        reverse_affinity[i, destinations] = np.exp(
            -np.maximum(reverse_values, 0.0) / (local_scale + eps)
        )

        finite_values = values[np.isfinite(values)]
        if finite_values.size > 1:
            spread = float(np.std(finite_values))
            contrast = spread / (float(np.mean(finite_values)) + eps)
            contrast = float(np.clip(contrast, 0.0, 1.0))
        else:
            contrast = 0.0

        if out_counts[i] <= 2:
            local_weight = 0.24 + 0.14 * contrast
            rank_weight = 0.34 - 0.08 * contrast
            regret_weight = 0.37
            reverse_weight = 0.05
        else:
            local_weight = 0.34 + 0.18 * contrast
            rank_weight = 0.31 - 0.11 * contrast
            regret_weight = 0.29
            reverse_weight = 0.06

        score = (
            local_weight * local_affinity[i, destinations]
            + rank_weight * bidirectional_rank[i, destinations]
            + regret_weight * regret[i, destinations]
            + reverse_weight * reverse_affinity[i, destinations]
        )

        result[i, destinations] = np.maximum(score, 0.0)

        maximum = float(np.max(result[i]))
        if np.isfinite(maximum) and maximum > 0.0:
            result[i] /= maximum

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)
    return result
