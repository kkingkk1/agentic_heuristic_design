import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    # Build a symmetric graph from the best available direction.
    safe_distances = np.minimum(
        np.where(directed_valid, distances, np.inf),
        np.where(directed_valid.T, distances.T, np.inf),
    )
    valid = np.isfinite(safe_distances) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    valid_values = safe_distances[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(valid_values)), 1.0)
    global_scale = max(global_scale, eps)

    degree = valid.sum(axis=1)
    candidate_k = min(max(3, int(np.sqrt(n)) + 1), n - 1)

    order = np.argsort(
        np.where(valid, safe_distances, np.inf),
        axis=1,
        kind="stable",
    )
    ranks = np.full((n, n), float(n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    nearest_indices = order[:, :candidate_k]
    nearest_distances = np.take_along_axis(
        safe_distances,
        nearest_indices,
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_distances)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_distances, 0.0),
        axis=1,
    ) / np.maximum(nearest_valid.sum(axis=1), 1)

    local_scale = np.where(
        degree > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    distance_ratio = safe_distances / np.maximum(pair_scale, eps)
    distance_term = 1.0 / (1.0 + np.clip(distance_ratio, 0.0, 50.0))

    endpoint_term = np.sqrt(
        1.0 / (
            1.0
            + np.clip(
                safe_distances / local_scale[:, None],
                0.0,
                50.0,
            )
        )
        * 1.0 / (
            1.0
            + np.clip(
                safe_distances / local_scale[None, :],
                0.0,
                50.0,
            )
        )
    )

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    rank_term = 0.35 + 0.65 * reciprocal_rank

    candidate_mask = valid & (ranks < float(candidate_k))
    mutual_candidate = (
        candidate_mask
        & candidate_mask.T
    ).astype(float)

    candidate_count = candidate_mask.sum(axis=1).astype(float)
    shared_candidates = candidate_mask.astype(float) @ candidate_mask.astype(float).T
    shared_term = 1.0 + 0.10 * np.minimum(
        shared_candidates / max(float(candidate_k), 1.0),
        1.0,
    )

    # Reward edges that are especially useful to at least one sparse endpoint.
    local_best = np.min(
        np.where(valid, safe_distances, np.inf),
        axis=1,
    )
    local_best = np.where(
        np.isfinite(local_best),
        local_best,
        global_scale,
    )

    endpoint_regret = np.minimum(
        safe_distances / np.maximum(local_best[:, None], eps),
        safe_distances / np.maximum(local_best[None, :], eps),
    )
    regret_term = 1.0 + 0.16 * np.clip(
        1.0 / (1.0 + np.maximum(endpoint_regret - 1.0, 0.0)),
        0.0,
        1.0,
    )

    # Preserve alternatives for nodes whose nearest options are unusually similar.
    sorted_nearest = np.where(
        np.isfinite(nearest_distances),
        nearest_distances,
        global_scale,
    )
    first_distance = sorted_nearest[:, 0]
    second_distance = sorted_nearest[:, min(1, candidate_k - 1)]
    ambiguity = second_distance / np.maximum(first_distance, eps)
    ambiguity = np.clip(ambiguity, 1.0, 3.0)

    ambiguity_term = 1.0 + 0.08 * np.minimum(
        np.maximum(ambiguity[:, None], ambiguity[None, :]) - 1.0,
        2.0,
    )

    # Mildly suppress edges incident to globally popular candidate endpoints.
    popularity = candidate_mask.astype(float).sum(axis=0)
    popularity_reference = max(float(np.mean(popularity)), 1.0)
    popularity_ratio = np.clip(
        popularity / popularity_reference,
        0.0,
        4.0,
    )
    popularity_term = 1.0 / np.sqrt(
        (1.0 + 0.08 * popularity_ratio[:, None])
        * (1.0 + 0.08 * popularity_ratio[None, :])
    )

    mutual_term = 1.0 + 0.20 * mutual_candidate

    scores = (
        distance_term
        * endpoint_term
        * rank_term
        * mutual_term
        * shared_term
        * regret_term
        * ambiguity_term
        * popularity_term
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
