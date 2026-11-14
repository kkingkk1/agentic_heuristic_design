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
    valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, distances, np.inf)
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
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    col_rank = rank.T

    valid_counts = valid.sum(axis=1)
    neighborhood_size = min(4, n - 1)

    neighborhood = np.take_along_axis(
        safe,
        order[:, :neighborhood_size],
        axis=1,
    )
    neighborhood_valid = np.isfinite(neighborhood)

    local_scale = np.divide(
        np.where(neighborhood_valid, neighborhood, 0.0).sum(axis=1),
        np.maximum(neighborhood_valid.sum(axis=1), 1),
    )
    local_scale = np.where(valid_counts > 0, local_scale, global_scale)
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
    )

    endpoint_best_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe / np.maximum(endpoint_best_scale, eps)
    opportunity_penalty = np.exp(
        -0.22 * np.clip(opportunity_ratio - 1.0, 0.0, 6.0)
    )

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )
    two_hop_bonus = 1.0 + 0.07 * np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
    ) * locally_competitive

    mutual_bonus = (
        1.0
        + 0.26 * mutual_nearest
        + 0.09 * mutual_top_neighbors
    )

    isolation_ratio = pair_scale / np.maximum(
        safe + global_scale,
        eps,
    )
    isolation_bonus = 1.0 + 0.13 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    edge_strength = np.where(
        valid,
        row_affinity / np.sqrt(1.0 + rank),
        0.0,
    )

    strongest = np.max(edge_strength, axis=1)
    total_strength = np.sum(edge_strength, axis=1)

    concentration = strongest / np.maximum(total_strength, eps)
    concentration = np.clip(concentration, 0.0, 1.0)

    sorted_strength = np.sort(edge_strength, axis=1)
    if n >= 2:
        second_best = sorted_strength[:, -2]
    else:
        second_best = np.zeros(n, dtype=float)

    alternative_ratio = second_best / np.maximum(strongest, eps)
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)

    sparsity = local_scale / np.maximum(global_scale, eps)
    sparsity = np.clip(sparsity, 0.0, 2.0)
    sparsity = sparsity / (1.0 + sparsity)

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + col_rank), 0.0, 1.0)
    )

    concentration_penalty = (
        np.sqrt(concentration[:, None] * concentration[None, :])
        * edge_topness
    )

    complementary_opportunity = np.sqrt(
        alternative_ratio[:, None] * alternative_ratio[None, :]
    )

    sparse_alternative_bonus = (
        np.sqrt(sparsity[:, None] * sparsity[None, :])
        * (1.0 - edge_topness)
    )

    base_scores = (
        distance_score
        * endpoint_affinity
        * (0.60 + 0.40 * reciprocal_rank)
        * mutual_bonus
        * opportunity_penalty
        * two_hop_bonus
        * isolation_bonus
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    log_scores = np.log(base_scores)

    log_scores += (
        0.065
        * complementary_opportunity
        * locally_competitive.astype(float)
    )

    log_scores += 0.045 * sparse_alternative_bonus
    log_scores -= 0.075 * concentration_penalty

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
