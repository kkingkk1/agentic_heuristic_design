import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
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

    valid_values = distance_matrix[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps

    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -distance_matrix[valid] / (global_scale + eps)
    )

    local_affinity = np.zeros((n, n), dtype=float)
    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)

    out_counts = np.sum(valid, axis=1).astype(float)
    in_counts = np.sum(valid, axis=0).astype(float)

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
            outgoing_rank[i, j] = 1.0 / (1.0 + float(rank))

        for j in destinations:
            alternatives = destinations[destinations != j]
            if alternatives.size:
                best_alternative = float(
                    np.min(distance_matrix[i, alternatives])
                )
                edge = float(distance_matrix[i, j])
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
                edge = float(distance_matrix[i, j])
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
        for rank, i in enumerate(order):
            incoming_rank[i, j] = 1.0 / (1.0 + float(rank))

    bidirectional_rank = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    score = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        degree = out_counts[i]

        if degree <= 2:
            local_weight = 0.30
            rank_weight = 0.35
            regret_weight = 0.30
            global_weight = 0.05
        else:
            local_weight = 0.38
            rank_weight = 0.30
            regret_weight = 0.25
            global_weight = 0.07

        score[i, row_valid] = (
            local_weight * local_affinity[i, row_valid]
            + rank_weight * bidirectional_rank[i, row_valid]
            + regret_weight * regret[i, row_valid]
            + global_weight * global_affinity[i, row_valid]
        )

    score[~valid] = 0.0

    for i in range(n):
        row = score[i]
        maximum = float(np.max(row))
        if maximum > 0.0 and np.isfinite(maximum):
            score[i] = row / maximum

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score
