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

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    valid_counts = np.sum(valid, axis=1)
    neighborhood_size = min(4, n - 1)

    neighborhood = np.take_along_axis(
        safe,
        row_order[:, :neighborhood_size],
        axis=1,
    )
    neighborhood_valid = np.isfinite(neighborhood)

    local_scale = np.sum(
        np.where(neighborhood_valid, neighborhood, 0.0),
        axis=1,
    ) / np.maximum(np.sum(neighborhood_valid, axis=1), 1)

    local_scale = np.where(
        valid_counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    row_affinity = np.exp(
        -np.clip(safe / local_scale[:, None], 0.0, 50.0)
    )
    col_affinity = np.exp(
        -np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )
    endpoint_affinity = np.sqrt(row_affinity * col_affinity)

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
    top_neighbors = valid & (row_rank < float(top_k))

    mutual_top_neighbors = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    competitive_k = min(4, n - 1)
    locally_competitive = (
        valid
        & (row_rank < float(competitive_k))
        & (col_rank < float(competitive_k))
    )

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_best, eps)

    # Keep the opportunity correction weak so moderately long,
    # structurally useful edges remain available.
    opportunity_penalty = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    shared_neighbor_count = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )
    shared_neighbor_signal = shared_neighbor_count / max(float(top_k), 1.0)

    validated_two_hop_bonus = (
        1.0
        + 0.10
        * np.clip(shared_neighbor_signal, 0.0, 1.0)
        * locally_competitive.astype(float)
    )

    mutual_bonus = (
        1.0
        + 0.38 * mutual_nearest
        + 0.16 * mutual_top_neighbors
    )

    isolation_bonus = 1.0 + 0.14 * np.clip(
        pair_scale / np.maximum(safe + global_scale, eps),
        0.0,
        1.0,
    )

    local_strength = np.where(
        valid,
        row_affinity / np.sqrt(1.0 + row_rank),
        0.0,
    )

    sorted_strength = np.sort(local_strength, axis=1)
    strongest_strength = sorted_strength[:, -1]
    second_strength = (
        sorted_strength[:, -2]
        if n >= 2
        else np.zeros(n, dtype=float)
    )

    alternative_ratio = second_strength / np.maximum(
        strongest_strength,
        eps,
    )
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)

    complementary_opportunity = np.sqrt(
        alternative_ratio[:, None] * alternative_ratio[None, :]
    )

    base_scores = (
        distance_score
        * endpoint_affinity
        * (0.54 + 0.46 * reciprocal_rank)
        * mutual_bonus
        * opportunity_penalty
        * validated_two_hop_bonus
        * isolation_bonus
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    log_scores = np.log(base_scores)
    log_scores += (
        0.09
        * complementary_opportunity
        * locally_competitive.astype(float)
    )

    scores = np.exp(np.clip(log_scores, -60.0, 30.0))
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
