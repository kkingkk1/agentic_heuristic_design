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
    raw_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(raw_valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(raw_valid, distances, np.inf)
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

    global_scale = max(global_scale, np.finfo(float).eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    usable = min(4, n - 1)
    nearest_distances = np.take_along_axis(
        safe,
        row_order[:, :usable],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_distances)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_distances, 0.0),
        axis=1,
    ) / np.maximum(nearest_valid.sum(axis=1), 1)

    local_scale = np.where(
        nearest_valid.sum(axis=1) > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = safe / np.maximum(pair_scale, np.finfo(float).eps)
    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 50.0))
    )

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    nearest_row = row_order[:, 0]
    nearest_col = col_order[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    mutual_top_neighbors = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    competitive_k = min(4, n - 1)
    locally_competitive = (
        valid
        & (row_rank < competitive_k)
        & (col_rank < competitive_k)
    )

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe / np.maximum(
        endpoint_best,
        np.finfo(float).eps,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.10 * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    shared_neighbor_count = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    union_count = (
        top_neighbors.sum(axis=1)[:, None]
        + top_neighbors.sum(axis=1)[None, :]
        - shared_neighbor_count
    )

    neighborhood_overlap = shared_neighbor_count / np.maximum(
        union_count,
        1.0,
    )

    complementary_coverage = (
        1.0
        + 0.12
        * np.clip(1.0 - neighborhood_overlap, 0.0, 1.0)
        * locally_competitive.astype(float)
    )

    validated_two_hop_bonus = (
        1.0
        + 0.07
        * np.clip(
            shared_neighbor_count / max(float(top_k), 1.0),
            0.0,
            1.0,
        )
        * locally_competitive.astype(float)
    )

    candidate_popularity = top_neighbors.astype(float).sum(axis=0)
    popularity_reference = max(
        float(np.mean(candidate_popularity)),
        1.0,
    )
    popularity_ratio = candidate_popularity / popularity_reference
    popularity_ratio = np.clip(popularity_ratio, 0.0, 3.0)

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + 0.09 * popularity_ratio[:, None])
        * (1.0 + 0.09 * popularity_ratio[None, :])
    )

    mutual_bonus = (
        1.0
        + 0.28 * mutual_nearest
        + 0.10 * mutual_top_neighbors
    )

    local_scale_ratio = pair_scale / (safe + global_scale)
    isolation_bonus = 1.0 + 0.10 * np.clip(
        local_scale_ratio,
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_affinity
        * (0.55 + 0.45 * reciprocal_rank)
        * mutual_bonus
        * opportunity_penalty
        * validated_two_hop_bonus
        * complementary_coverage
        * concentration_penalty
        * isolation_bonus
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
