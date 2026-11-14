import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)

    # An undirected edge is valid only when both directed entries are valid.
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal

    if not np.any(valid):
        return score

    symmetric_distance = 0.5 * (distances + distances.T)
    valid_values = symmetric_distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps

    # Direct global efficiency; zero-distance edges receive affinity 1.
    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -symmetric_distance[valid] / (scale + eps)
    )

    # Endpoint-local affinity and rank evidence.
    local_affinity = np.zeros((n, n), dtype=float)
    rank_affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        row_distances = symmetric_distance[i, neighbors]
        positive_local = row_distances[row_distances > 0.0]

        if positive_local.size:
            local_scale = float(np.median(
                np.sort(positive_local)[:min(3, positive_local.size)]
            ))
        else:
            local_scale = scale

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = scale

        order = neighbors[np.argsort(symmetric_distance[i, neighbors])]

        for rank, j in enumerate(order):
            rank_affinity[i, j] = 1.0 / (1.0 + rank)

        local_affinity[i, neighbors] = np.exp(
            -symmetric_distance[i, neighbors] / (local_scale + eps)
        )

    symmetric_local = np.sqrt(
        np.maximum(local_affinity, 0.0)
        * np.maximum(local_affinity.T, 0.0)
    )
    symmetric_rank = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )

    neighborhood_signal = 0.75 * symmetric_local + 0.25 * symmetric_rank

    # Endpoint competition: high when the candidate is much better than
    # the best alternative available at either endpoint.
    competition = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors_i = np.flatnonzero(valid[i])

        for j in neighbors_i:
            alternatives_i = neighbors_i[neighbors_i != j]
            alternatives_j = np.flatnonzero(valid[:, j])
            alternatives_j = alternatives_j[alternatives_j != i]

            if alternatives_i.size == 0 or alternatives_j.size == 0:
                continue

            best_i = float(np.min(symmetric_distance[i, alternatives_i]))
            best_j = float(np.min(symmetric_distance[alternatives_j, j]))
            candidate = float(symmetric_distance[i, j])

            pressure_i = (best_i - candidate) / (best_i + candidate + eps)
            pressure_j = (best_j - candidate) / (best_j + candidate + eps)

            competition[i, j] = max(0.0, 0.5 * (pressure_i + pressure_j))

    competition = 0.5 * (competition + competition.T)

    # Best valid two-hop detour, using only mutually valid edges.
    detour_advantage = np.zeros((n, n), dtype=float)

    if n >= 3:
        working = np.where(valid, symmetric_distance, np.inf)
        two_hop = np.full((n, n), np.inf, dtype=float)

        for k in range(n):
            two_hop = np.minimum(
                two_hop,
                working[:, k:k + 1] + working[k:k + 1, :]
            )

        finite_detour = valid & np.isfinite(two_hop)
        detour_advantage[finite_detour] = np.clip(
            (
                two_hop[finite_detour]
                - symmetric_distance[finite_detour]
            )
            / (
                two_hop[finite_detour]
                + symmetric_distance[finite_detour]
                + eps
            ),
            0.0,
            1.0,
        )

    raw_score = (
        0.35 * global_affinity
        + 0.25 * neighborhood_signal
        + 0.20 * competition
        + 0.20 * detour_advantage
    )

    raw_score[~valid] = 0.0
    raw_score = 0.5 * (raw_score + raw_score.T)
    np.fill_diagonal(raw_score, 0.0)

    maximum = float(np.max(raw_score))
    if np.isfinite(maximum) and maximum > 0.0:
        raw_score /= maximum

    return raw_score.astype(float, copy=False)
