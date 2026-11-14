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

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal

    if not np.any(valid):
        return result

    symmetric_distance = 0.5 * (
        distance_matrix + distance_matrix.T
    )

    valid_values = symmetric_distance[valid]
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
        -symmetric_distance[valid] / (global_scale + eps)
    )

    local_affinity = np.zeros((n, n), dtype=float)
    rank_affinity = np.zeros((n, n), dtype=float)
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        row_values = symmetric_distance[i, neighbors]
        ordered_values = np.sort(row_values)
        positive_local = ordered_values[ordered_values > 0.0]

        if positive_local.size:
            scale = float(np.median(positive_local[: min(3, positive_local.size)]))
        else:
            scale = global_scale

        if not np.isfinite(scale) or scale <= 0.0:
            scale = global_scale

        local_scale[i] = scale

        order = neighbors[np.argsort(row_values)]
        for rank, j in enumerate(order):
            rank_affinity[i, j] = 1.0 / (1.0 + float(rank))

        local_affinity[i, neighbors] = np.exp(
            -symmetric_distance[i, neighbors] / (scale + eps)
        )

    symmetric_local = np.sqrt(
        np.maximum(local_affinity, 0.0)
        * np.maximum(local_affinity.T, 0.0)
    )
    symmetric_rank = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )

    base_affinity = (
        0.42 * global_affinity
        + 0.38 * symmetric_local
        + 0.20 * symmetric_rank
    )

    savings = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors_i = np.flatnonzero(valid[i])

        for j in neighbors_i:
            alternatives_i = neighbors_i[neighbors_i != j]
            neighbors_j = np.flatnonzero(valid[:, j])
            alternatives_j = neighbors_j[neighbors_j != i]

            if alternatives_i.size == 0 or alternatives_j.size == 0:
                continue

            alternative_i = float(
                np.min(symmetric_distance[i, alternatives_i])
            )
            alternative_j = float(
                np.min(symmetric_distance[alternatives_j, j])
            )
            candidate = float(symmetric_distance[i, j])

            denominator = alternative_i + alternative_j + eps
            saving = (
                alternative_i + alternative_j - 2.0 * candidate
            ) / denominator

            savings[i, j] = max(saving, 0.0)

    savings = 0.5 * (savings + savings.T)
    positive_savings = savings[valid & (savings > 0.0)]

    if positive_savings.size:
        savings_scale = float(np.percentile(positive_savings, 90))
        if not np.isfinite(savings_scale) or savings_scale <= 0.0:
            savings_scale = float(np.max(positive_savings))

        normalized_savings = np.clip(
            savings / (savings_scale + eps),
            0.0,
            1.0,
        )
    else:
        normalized_savings = np.zeros_like(savings)

    base_score = (
        0.46 * base_affinity
        + 0.16 * normalized_savings
    )

    # Directed detour signal:
    # an edge is useful when its direct cost is lower than the best
    # two-hop route connecting the same endpoints.
    detour = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors_i = np.flatnonzero(valid[i])

        for j in neighbors_i:
            intermediates = np.flatnonzero(valid[i] & valid[:, j])
            intermediates = intermediates[
                (intermediates != i) & (intermediates != j)
            ]

            if intermediates.size == 0:
                continue

            two_hop = symmetric_distance[i, intermediates]
            two_hop = two_hop + symmetric_distance[intermediates, j]
            best_detour = float(np.min(two_hop))
            direct = float(symmetric_distance[i, j])

            denominator = best_detour + eps
            detour[i, j] = np.clip(
                (best_detour - direct) / denominator,
                0.0,
                1.0,
            )

    detour = 0.5 * (detour + detour.T)

    # Endpoint bottleneck signal:
    # favor edges whose endpoints have few comparably strong alternatives.
    bottleneck = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors_i = np.flatnonzero(valid[i])
        if neighbors_i.size == 0:
            continue

        row_distances = symmetric_distance[i, neighbors_i]
        best_i = float(np.min(row_distances))
        tolerance_i = best_i + max(local_scale[i], eps)
        flexibility_i = np.sum(row_distances <= tolerance_i)
        scarcity_i = 1.0 / max(float(flexibility_i), 1.0)

        for j in neighbors_i:
            neighbors_j = np.flatnonzero(valid[:, j])
            if neighbors_j.size == 0:
                continue

            column_distances = symmetric_distance[neighbors_j, j]
            best_j = float(np.min(column_distances))
            tolerance_j = best_j + max(local_scale[j], eps)
            flexibility_j = np.sum(column_distances <= tolerance_j)
            scarcity_j = 1.0 / max(float(flexibility_j), 1.0)

            bottleneck[i, j] = np.sqrt(scarcity_i * scarcity_j)

    bottleneck = np.clip(
        0.5 * (bottleneck + bottleneck.T),
        0.0,
        1.0,
    )

    # Add the new signals only where the base heuristic already provides
    # meaningful evidence, limiting amplification of anomalous edges.
    base_gate = np.clip((base_score - 0.15) / 0.55, 0.0, 1.0)
    selective_detour = detour * base_gate
    selective_bottleneck = bottleneck * base_gate

    score = (
        base_score
        + 0.10 * selective_detour
        + 0.08 * selective_bottleneck
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    finite_score = np.isfinite(score)
    score[~finite_score] = 0.0

    return np.clip(score, 0.0, 1.0)
