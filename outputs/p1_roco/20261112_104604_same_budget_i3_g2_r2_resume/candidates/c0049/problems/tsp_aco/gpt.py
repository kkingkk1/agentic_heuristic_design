import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)

    # Only mutually valid edges are usable in the undirected TSP heuristic.
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    symmetric_distance = 0.5 * (distance_matrix + distance_matrix.T)

    # Endpoint-local scales avoid global scaling and preserve local contrast.
    local_scale = np.ones(n, dtype=float)
    local_rank = np.zeros((n, n), dtype=float)
    local_affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        values = symmetric_distance[i, neighbors]
        order = np.argsort(values, kind="stable")
        ordered_neighbors = neighbors[order]
        ordered_values = values[order]

        positive = ordered_values[ordered_values > 0.0]
        if positive.size:
            scale = float(np.median(positive[: min(4, positive.size)]))
        else:
            scale = 1.0

        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        local_scale[i] = scale

        for rank, j in enumerate(ordered_neighbors):
            local_rank[i, j] = float(rank)
            local_affinity[i, j] = np.exp(
                -symmetric_distance[i, j] / (scale + eps)
            )

    # Symmetric local evidence forms the base heuristic B.
    symmetric_affinity = np.sqrt(
        np.maximum(local_affinity, 0.0)
        * np.maximum(local_affinity.T, 0.0)
    )

    rank_affinity = 1.0 / (1.0 + local_rank)
    symmetric_rank_affinity = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )

    base = (
        0.62 * symmetric_affinity
        + 0.23 * symmetric_rank_affinity
    )

    # Correct two-sided local savings.
    savings = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors_i = np.flatnonzero(valid[i])

        for j in neighbors_i:
            alternatives_i = neighbors_i[neighbors_i != j]
            alternatives_j = np.flatnonzero(valid[:, j])
            alternatives_j = alternatives_j[alternatives_j != i]

            if alternatives_i.size == 0 or alternatives_j.size == 0:
                continue

            alt_out = float(np.min(distance_matrix[i, alternatives_i]))
            alt_in = float(np.min(distance_matrix[alternatives_j, j]))
            candidate = float(symmetric_distance[i, j])

            denominator = alt_out + alt_in + eps
            value = (alt_out + alt_in - 2.0 * candidate) / denominator
            savings[i, j] = max(value, 0.0)

    savings = 0.5 * (savings + savings.T)

    positive_savings = savings[valid & (savings > 0.0)]
    if positive_savings.size:
        reference = float(np.percentile(positive_savings, 85))
        if not np.isfinite(reference) or reference <= 0.0:
            reference = float(np.max(positive_savings))
        normalized_savings = np.clip(
            savings / (reference + eps), 0.0, 1.0
        )
    else:
        normalized_savings = np.zeros_like(savings)

    base = base + 0.15 * normalized_savings

    # Directed-detour signal:
    # compare each directed edge against the best alternative outgoing
    # edge at its source and incoming edge at its destination.
    directed_detour = np.zeros((n, n), dtype=float)

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])
        if outgoing.size == 0:
            continue

        for j in outgoing:
            alternatives_out = outgoing[outgoing != j]
            incoming = np.flatnonzero(valid[:, j])
            alternatives_in = incoming[incoming != i]

            if alternatives_out.size == 0 or alternatives_in.size == 0:
                continue

            best_out = float(np.min(distance_matrix[i, alternatives_out]))
            best_in = float(np.min(distance_matrix[alternatives_in, j]))
            candidate = float(distance_matrix[i, j])

            denominator = best_out + best_in + eps
            detour = (best_out + best_in - 2.0 * candidate) / denominator
            directed_detour[i, j] = max(detour, 0.0)

    detour = 0.5 * (directed_detour + directed_detour.T)

    positive_detour = detour[valid & (detour > 0.0)]
    if positive_detour.size:
        detour_reference = float(np.percentile(positive_detour, 85))
        if not np.isfinite(detour_reference) or detour_reference <= 0.0:
            detour_reference = float(np.max(positive_detour))
        detour = np.clip(detour / (detour_reference + eps), 0.0, 1.0)

    # Endpoint bottleneck signal:
    # an edge is valuable when each endpoint has few alternatives within
    # a modest local distance band.
    competitive_count = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        for j in neighbors:
            threshold = 1.25 * max(
                symmetric_distance[i, j],
                local_scale[i] * 0.25,
            )
            competitive_count[i, j] = float(
                np.count_nonzero(
                    symmetric_distance[i, neighbors] <= threshold
                )
            )

    endpoint_bottleneck = np.zeros((n, n), dtype=float)
    endpoint_bottleneck[valid] = 1.0 / np.sqrt(
        np.maximum(competitive_count[valid], 1.0)
        * np.maximum(competitive_count.T[valid], 1.0)
    )

    bottleneck_values = endpoint_bottleneck[valid]
    if bottleneck_values.size:
        bottleneck_min = float(np.min(bottleneck_values))
        bottleneck_max = float(np.max(bottleneck_values))
        if bottleneck_max > bottleneck_min:
            endpoint_bottleneck[valid] = (
                endpoint_bottleneck[valid] - bottleneck_min
            ) / (bottleneck_max - bottleneck_min)

    # Small coverage regularizer. It rewards edges whose endpoint ranks are
    # jointly good without performing column normalization.
    coverage = np.zeros((n, n), dtype=float)
    coverage[valid] = 1.0 / (
        1.0
        + np.maximum(local_rank[valid], 0.0)
        + np.maximum(local_rank.T[valid], 0.0)
    )

    coverage_values = coverage[valid]
    if coverage_values.size:
        coverage_scale = float(np.percentile(coverage_values, 90))
        if not np.isfinite(coverage_scale) or coverage_scale <= 0.0:
            coverage_scale = float(np.max(coverage_values))
        coverage = np.clip(coverage / (coverage_scale + eps), 0.0, 1.0)

    score = (
        base
        + 0.08 * detour
        + 0.08 * endpoint_bottleneck
        + 0.04 * coverage
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.maximum(score, 0.0)
