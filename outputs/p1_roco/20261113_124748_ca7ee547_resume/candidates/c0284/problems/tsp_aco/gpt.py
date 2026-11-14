import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid_input = (
        np.isfinite(distances)
        & (distances >= 0.0)
    )
    np.fill_diagonal(valid_input, False)

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, np.finfo(float).eps)

    neighbor_count = min(5, n - 1)
    order = np.argsort(safe, axis=1, kind="stable")
    nearest = np.take_along_axis(
        safe,
        order[:, :neighbor_count],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest)

    local_scale = np.divide(
        np.sum(np.where(nearest_valid, nearest, 0.0), axis=1),
        np.maximum(nearest_valid.sum(axis=1), 1),
    )
    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    edge_distance = safe / np.maximum(pair_scale, np.finfo(float).eps)

    distance_affinity = np.exp(-np.clip(edge_distance, 0.0, 40.0))

    rank = np.empty((n, n), dtype=float)
    rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    rank_affinity = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    candidate_k = min(4, n - 1)
    candidate = np.zeros((n, n), dtype=bool)
    candidate[
        np.arange(n)[:, None],
        order[:, :candidate_k],
    ] = True
    candidate &= valid

    mutual_candidate = candidate & candidate.T

    endpoint_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    relative_cost = safe / np.maximum(endpoint_scale, np.finfo(float).eps)

    regret = np.exp(
        -np.clip(np.maximum(relative_cost - 1.0, 0.0), 0.0, 12.0)
    )
    regret = 0.75 + 0.25 * regret

    candidate_popularity = candidate.astype(float).sum(axis=0)
    popularity_reference = max(
        float(np.mean(candidate_popularity)),
        1.0,
    )
    popularity = np.clip(
        candidate_popularity / popularity_reference,
        0.0,
        3.0,
    )

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + 0.16 * popularity[:, None])
        * (1.0 + 0.16 * popularity[None, :])
    )

    candidate_float = candidate.astype(float)
    shared_neighbors = candidate_float @ candidate_float.T
    candidate_sizes = candidate_float.sum(axis=1)

    union_size = (
        candidate_sizes[:, None]
        + candidate_sizes[None, :]
        - shared_neighbors
    )
    overlap = shared_neighbors / np.maximum(union_size, 1.0)

    complementary_bonus = 1.0 + (
        0.12
        * (1.0 - np.clip(overlap, 0.0, 1.0))
        * mutual_candidate.astype(float)
    )

    mutual_bonus = (
        1.0
        + 0.18 * mutual_candidate.astype(float)
    )

    local_density = np.divide(
        nearest[:, : min(3, neighbor_count)].sum(axis=1),
        np.maximum(
            nearest_valid[:, : min(3, neighbor_count)].sum(axis=1),
            1,
        ),
    )
    local_density = np.where(
        np.isfinite(local_density),
        local_density,
        global_scale,
    )

    density_ratio = np.sqrt(
        local_density[:, None] * local_density[None, :]
    ) / np.maximum(safe + global_scale, np.finfo(float).eps)
    density_bonus = 1.0 + 0.10 * np.clip(density_ratio, 0.0, 1.0)

    scores = (
        distance_affinity
        * (0.52 + 0.48 * rank_affinity)
        * regret
        * mutual_bonus
        * complementary_bonus
        * concentration_penalty
        * density_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
