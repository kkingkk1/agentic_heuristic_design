import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)

    # Undirected TSP scoring requires both directions to be valid.
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal

    if not np.any(valid):
        return score

    symmetric_distance = 0.5 * (distances + distances.T)
    eps = np.finfo(float).eps

    valid_values = symmetric_distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Global direct-edge affinity.
    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -symmetric_distance[valid] / (scale + eps)
    )

    # Endpoint-local affinity and rank-based neighborhood evidence.
    local_affinity = np.zeros((n, n), dtype=float)
    rank_affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])

        if neighbors.size == 0:
            continue

        row_values = symmetric_distance[i, neighbors]
        ordered_values = np.sort(row_values)

        nearest_count = min(3, ordered_values.size)
        nearest_values = ordered_values[:nearest_count]
        positive_nearest = nearest_values[nearest_values > 0.0]

        if positive_nearest.size:
            local_scale = float(np.median(positive_nearest))
        else:
            local_scale = scale

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = scale

        local_affinity[i, neighbors] = np.exp(
            -row_values / (local_scale + eps)
        )

        order = neighbors[np.argsort(row_values, kind="mergesort")]
        for rank, j in enumerate(order):
            rank_affinity[i, j] = 1.0 / (1.0 + float(rank))

    symmetric_local_affinity = np.sqrt(
        np.maximum(local_affinity, 0.0)
        * np.maximum(local_affinity.T, 0.0)
    )

    symmetric_rank_affinity = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )

    neighborhood_signal = (
        0.7 * symmetric_local_affinity
        + 0.3 * symmetric_rank_affinity
    )

    # Compute the best valid two-hop detour for every candidate edge.
    work = np.where(valid, symmetric_distance, np.inf)
    two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        candidate = work[:, k:k + 1] + work[k:k + 1, :]
        two_hop = np.minimum(two_hop, candidate)

    np.fill_diagonal(two_hop, np.inf)

    detour_advantage = np.zeros((n, n), dtype=float)
    finite_detours = valid & np.isfinite(two_hop)

    denominator = two_hop + symmetric_distance + eps
    detour_advantage[finite_detours] = (
        two_hop[finite_detours] - symmetric_distance[finite_detours]
    ) / denominator[finite_detours]
    detour_advantage = np.clip(detour_advantage, 0.0, 1.0)

    # Measure endpoint bottleneck pressure against the best alternative edge.
    endpoint_pressure = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])

        for j in neighbors:
            alternatives_i = neighbors[neighbors != j]
            alternatives_j = np.flatnonzero(valid[:, j])
            alternatives_j = alternatives_j[alternatives_j != i]

            if alternatives_i.size == 0 or alternatives_j.size == 0:
                continue

            best_i = float(np.min(symmetric_distance[i, alternatives_i]))
            best_j = float(np.min(symmetric_distance[alternatives_j, j]))
            candidate = float(symmetric_distance[i, j])

            pressure_i = (best_i - candidate) / (best_i + candidate + eps)
            pressure_j = (best_j - candidate) / (best_j + candidate + eps)

            endpoint_pressure[i, j] = np.sqrt(
                max(pressure_i, 0.0) * max(pressure_j, 0.0)
            )

    endpoint_pressure = 0.5 * (
        endpoint_pressure + endpoint_pressure.T
    )

    score = (
        0.30 * global_affinity
        + 0.25 * neighborhood_signal
        + 0.30 * detour_advantage
        + 0.15 * endpoint_pressure
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
