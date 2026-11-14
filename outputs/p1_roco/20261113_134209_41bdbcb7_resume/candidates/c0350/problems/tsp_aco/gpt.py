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

    masked = np.where(valid, safe, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(masked, order, axis=1)

    rank = np.full((n, n), float(n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    col_rank = rank.T

    valid_counts = valid.sum(axis=1)
    neighborhood_size = min(5, n - 1)
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
    normalized_distance = np.nan_to_num(
        normalized_distance,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )

    row_rank_score = 1.0 / np.sqrt(1.0 + rank)
    symmetric_rank_score = np.sqrt(row_rank_score * row_rank_score.T)

    candidate_k = min(3, n - 1)
    candidate = valid & (rank < float(candidate_k))
    reciprocal_candidate = candidate & candidate.T

    nearest = order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )

    competitive_k = min(5, n - 1)
    locally_competitive = (
        valid
        & (rank < float(competitive_k))
        & (col_rank < float(competitive_k))
    )

    best_distance = np.min(masked, axis=1)
    second_distance = np.full(n, np.inf, dtype=float)

    if n > 1:
        second_distance = np.partition(masked, 1, axis=1)[:, 1]

    best_distance = np.where(np.isfinite(best_distance), best_distance, global_scale)
    second_distance = np.where(
        np.isfinite(second_distance),
        second_distance,
        best_distance + global_scale,
    )

    alternative_gap = np.maximum(second_distance - best_distance, 0.0)
    alternative_strength = alternative_gap / np.maximum(
        second_distance + global_scale,
        eps,
    )

    endpoint_opportunity = np.minimum(
        alternative_strength[:, None],
        alternative_strength[None, :],
    )

    edge_opportunity = safe / np.maximum(
        np.minimum(best_distance[:, None], best_distance[None, :]) + global_scale,
        eps,
    )
    opportunity_penalty = np.exp(
        -0.18 * np.clip(edge_opportunity - 0.25, 0.0, 8.0)
    )

    shared_neighbors = candidate.astype(float) @ candidate.astype(float).T
    diversity_bonus = 1.0 + 0.08 * np.clip(
        shared_neighbors / max(float(candidate_k), 1.0),
        0.0,
        1.0,
    ) * locally_competitive

    mutual_bonus = (
        1.0
        + 0.22 * mutual_nearest.astype(float)
        + 0.10 * reciprocal_candidate.astype(float)
    )

    isolation_ratio = pair_scale / np.maximum(safe + global_scale, eps)
    isolation_bonus = 1.0 + 0.10 * np.clip(isolation_ratio, 0.0, 1.0)

    base_log_score = (
        -1.05 * np.log1p(np.clip(normalized_distance, 0.0, 1e12))
        + 0.24 * np.log(np.maximum(symmetric_rank_score, eps))
        + np.log(np.maximum(mutual_bonus, eps))
        + np.log(np.maximum(opportunity_penalty, eps))
        + np.log(np.maximum(diversity_bonus, eps))
        + np.log(np.maximum(isolation_bonus, eps))
    )

    base_log_score += (
        0.12
        * endpoint_opportunity
        * locally_competitive.astype(float)
    )

    base_log_score[~valid] = -np.inf

    maximum_log_score = np.max(base_log_score)
    if not np.isfinite(maximum_log_score):
        return np.zeros((n, n), dtype=float)

    scores = np.exp(np.clip(base_log_score - maximum_log_score, -60.0, 0.0))
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
