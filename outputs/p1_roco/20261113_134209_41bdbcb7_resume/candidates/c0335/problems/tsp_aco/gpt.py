import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return scores

    raw = np.where(raw_valid, distances, np.inf)

    # Construct an undirected graph using the available directed value.
    safe = np.minimum(raw, raw.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return scores

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    valid_counts = valid.sum(axis=1)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.35 * np.sqrt(n)))),
    )

    candidate_mask = valid & (ranks < float(neighborhood_size))
    nearest_mask = valid & (ranks == 0.0)

    candidate_distances = np.take_along_axis(
        safe,
        order[:, :neighborhood_size],
        axis=1,
    )
    candidate_valid = np.isfinite(candidate_distances)

    local_scale = np.sum(
        np.where(candidate_valid, candidate_distances, 0.0),
        axis=1,
    ) / np.maximum(candidate_valid.sum(axis=1), 1)

    local_scale = np.where(
        valid_counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.05 * global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = np.divide(
        safe,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    endpoint_ratio_i = np.divide(
        safe,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio_j = np.divide(
        safe,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    endpoint_ratio_i = np.clip(endpoint_ratio_i, 0.0, 50.0)
    endpoint_ratio_j = np.clip(endpoint_ratio_j, 0.0, 50.0)

    distance_score = np.exp(
        -np.clip(normalized_distance ** 1.15, 0.0, 50.0)
    )

    endpoint_affinity = np.sqrt(
        np.exp(-endpoint_ratio_i)
        * np.exp(-endpoint_ratio_j)
    )

    rank_score = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        logits = (
            -1.05 * normalized_distance[i, row_valid]
            -0.10 * endpoint_ratio_i[i, row_valid]
        )
        logits -= np.max(logits)

        weights = np.exp(logits)
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    mutual_candidate = (
        candidate_mask
        & candidate_mask.T
        & valid
    ).astype(float)

    mutual_nearest = (
        nearest_mask
        & nearest_mask.T
        & valid
    ).astype(float)

    shared_neighbors = (
        candidate_mask.astype(float)
        @ candidate_mask.astype(float).T
    )
    shared_neighbor_signal = np.clip(
        shared_neighbors / max(float(neighborhood_size), 1.0),
        0.0,
        1.0,
    )

    # Nodes with few alternatives receive a bounded reserve-capacity bonus.
    second_distance = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_order = order[i]
        row_valid = valid[i, row_order]
        values = safe[i, row_order][row_valid]

        if values.size == 0:
            continue

        second_distance[i] = (
            values[1]
            if values.size > 1
            else values[0]
        )

    second_pair = np.sqrt(
        np.maximum(second_distance[:, None], eps)
        * np.maximum(second_distance[None, :], eps)
    )

    scarcity_ratio = np.divide(
        second_pair,
        safe + 0.5 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    scarcity_bonus = 1.0 + 0.16 * np.clip(
        scarcity_ratio - 0.75,
        0.0,
        3.0,
    )

    opportunity_ratio = np.divide(
        safe,
        np.maximum(
            np.minimum(
                local_scale[:, None],
                local_scale[None, :],
            ),
            eps,
        ),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.10 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    popularity = candidate_mask.astype(float).sum(axis=0)
    popularity_reference = max(float(np.mean(popularity)), 1.0)
    popularity_ratio = np.clip(
        popularity / popularity_reference,
        0.0,
        3.0,
    )

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + 0.10 * popularity_ratio[:, None])
        * (1.0 + 0.10 * popularity_ratio[None, :])
    )

    coverage_bonus = 1.0 + 0.08 * (
        shared_neighbor_signal
        * mutual_candidate
    )

    scores = (
        distance_score
        * endpoint_affinity
        * mutual_preference
        * (0.46 + 0.54 * rank_score)
        * (0.76 + 0.24 * np.maximum(
            preferences,
            preferences.T,
        ))
        * (1.0 + 0.58 * mutual_candidate)
        * (1.0 + 0.48 * mutual_nearest)
        * scarcity_bonus
        * opportunity_penalty
        * concentration_penalty
        * coverage_bonus
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
