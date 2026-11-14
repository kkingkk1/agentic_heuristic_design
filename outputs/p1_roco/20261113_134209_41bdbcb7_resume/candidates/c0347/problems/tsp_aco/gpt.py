import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid_input = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(safe, order, axis=1)
    sorted_valid = np.isfinite(sorted_distances)

    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    col_rank = rank.T

    valid_counts = valid.sum(axis=1)
    neighborhood_size = min(4, n - 1)

    neighborhood = sorted_distances[:, :neighborhood_size]
    neighborhood_valid = np.isfinite(neighborhood)

    local_scale = np.divide(
        np.where(neighborhood_valid, neighborhood, 0.0).sum(axis=1),
        np.maximum(neighborhood_valid.sum(axis=1), 1),
    )
    local_scale = np.where(valid_counts > 0, local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    row_affinity = np.exp(
        -np.clip(safe / local_scale[:, None], 0.0, 50.0)
    )
    col_affinity = np.exp(
        -np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )
    endpoint_affinity = np.sqrt(row_affinity * col_affinity)

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + col_rank)
    )

    top_k = min(2, n - 1)
    top_neighbors = valid & (rank < float(top_k))

    mutual_top_neighbors = (
        top_neighbors & top_neighbors.T
    ).astype(float)

    nearest = order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    competitive_k = min(4, n - 1)
    locally_competitive = (
        valid
        & (rank < float(competitive_k))
        & (col_rank < float(competitive_k))
    ).astype(float)

    best_distance = np.min(
        np.where(valid, safe, np.inf),
        axis=1,
    )

    if n >= 2:
        masked_for_second = np.where(valid, safe, np.inf)
        partitioned = np.partition(masked_for_second, 1, axis=1)
        second_distance = partitioned[:, 1]
    else:
        second_distance = best_distance.copy()

    second_distance = np.where(
        np.isfinite(second_distance),
        second_distance,
        best_distance + global_scale,
    )

    endpoint_regret = (
        (second_distance - best_distance)
        / np.maximum(second_distance + local_scale, eps)
    )
    endpoint_regret = np.clip(endpoint_regret, 0.0, 1.0)

    opportunity_ratio = safe / np.maximum(
        np.minimum(local_scale[:, None], local_scale[None, :]),
        eps,
    )
    opportunity_penalty = np.exp(
        -0.20 * np.clip(opportunity_ratio - 1.0, 0.0, 6.0)
    )

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )
    shared_neighbor_bonus = 1.0 + 0.06 * np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
    ) * locally_competitive

    mutual_bonus = (
        1.0
        + 0.22 * mutual_nearest
        + 0.08 * mutual_top_neighbors
    )

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + col_rank), 0.0, 1.0)
    )

    regret_bonus = 1.0 + 0.18 * np.sqrt(
        endpoint_regret[:, None] * endpoint_regret[None, :]
    ) * (0.5 + 0.5 * edge_topness)

    isolation_ratio = pair_scale / np.maximum(
        safe + global_scale,
        eps,
    )
    isolation_bonus = 1.0 + 0.10 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_affinity
        * (0.58 + 0.42 * reciprocal_rank)
        * mutual_bonus
        * opportunity_penalty
        * shared_neighbor_bonus
        * regret_bonus
        * isolation_bonus
    )

    scores[~valid] = 0.0
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
