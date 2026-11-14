import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized prior for stochastic TSP edge sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return result

    forward = np.where(raw_valid, distances, np.inf)
    backward = np.where(raw_valid.T, distances.T, np.inf)

    safe = np.minimum(forward, backward)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return result

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(safe, order, axis=1)

    ranks = np.full((n, n), float(n), dtype=float)
    top_k = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)
    local_k = min(4, n - 1)

    top_neighbors = np.zeros((n, n), dtype=bool)
    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    for i in range(n):
        finite_positions = np.flatnonzero(np.isfinite(sorted_distances[i]))
        if finite_positions.size == 0:
            continue

        neighbors = order[i, finite_positions]
        values = sorted_distances[i, finite_positions]

        ranks[i, neighbors] = np.arange(
            neighbors.size,
            dtype=float,
        )

        top_neighbors[i, neighbors[:top_k]] = True

        local_values = values[:local_k]
        positive_local = local_values[local_values > 0.0]

        if positive_local.size:
            local_scale[i] = float(np.mean(positive_local))
        elif local_values.size:
            local_scale[i] = float(np.mean(local_values))

        if values.size > 1:
            second_distance[i] = float(values[1])
        elif values.size == 1:
            second_distance[i] = float(values[0])

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized_distance = safe / np.maximum(pair_scale, eps)
        distance_score = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0)
        )

        endpoint_score = np.sqrt(
            np.exp(
                -np.clip(
                    safe / local_scale[:, None],
                    0.0,
                    50.0,
                )
            )
            * np.exp(
                -np.clip(
                    safe / local_scale[None, :],
                    0.0,
                    50.0,
                )
            )
        )

    rank_score = np.zeros((n, n), dtype=float)
    finite_rank = np.isfinite(ranks)

    rank_score[finite_rank] = 1.0 / (
        1.0 + ranks[finite_rank]
    )

    reciprocal_rank = np.sqrt(
        rank_score * rank_score.T
    )

    mutual_nearest = (
        valid
        & (ranks == 0.0)
        & (ranks.T == 0.0)
    ).astype(float)

    mutual_top = (
        valid
        & top_neighbors
        & top_neighbors.T
    ).astype(float)

    competitive_k = min(4, n - 1)
    competitive = (
        valid
        & (ranks < float(competitive_k))
        & (ranks.T < float(competitive_k))
    ).astype(float)

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe / np.maximum(
        endpoint_best,
        eps,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.16
        * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    scarcity_ratio = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    ) / np.maximum(safe, eps)

    scarcity_bonus = 1.0 + 0.10 * np.clip(
        scarcity_ratio - 1.0,
        0.0,
        4.0,
    )

    isolation_ratio = pair_scale / np.maximum(
        safe + global_scale,
        eps,
    )

    isolation_bonus = 1.0 + 0.14 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    neighborhood_sizes = top_neighbors.sum(axis=1).astype(float)

    union_sizes = (
        neighborhood_sizes[:, None]
        + neighborhood_sizes[None, :]
        - shared_neighbors
    )

    overlap = shared_neighbors / np.maximum(
        union_sizes,
        1.0,
    )

    local_strength = np.where(
        valid,
        np.exp(
            -np.clip(
                safe / local_scale[:, None],
                0.0,
                50.0,
            )
        )
        / np.sqrt(1.0 + ranks),
        0.0,
    )

    strongest = np.max(local_strength, axis=1)
    total_strength = np.sum(local_strength, axis=1)

    concentration = strongest / np.maximum(
        total_strength,
        eps,
    )
    concentration = np.clip(concentration, 0.0, 1.0)

    sorted_strength = np.sort(local_strength, axis=1)

    if n >= 2:
        second_strength = sorted_strength[:, -2]
    else:
        second_strength = np.zeros(n, dtype=float)

    alternative_ratio = second_strength / np.maximum(
        strongest,
        eps,
    )
    alternative_ratio = np.clip(
        alternative_ratio,
        0.0,
        1.0,
    )

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + ranks), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + ranks.T), 0.0, 1.0)
    )

    concentration_pressure = (
        np.sqrt(
            concentration[:, None]
            * concentration[None, :]
        )
        * edge_topness
    )

    complementary_opportunity = np.sqrt(
        alternative_ratio[:, None]
        * alternative_ratio[None, :]
    )

    local_sparsity = np.clip(
        local_scale / np.maximum(global_scale, eps),
        0.0,
        2.0,
    )
    local_sparsity = local_sparsity / (1.0 + local_sparsity)

    sparse_alternative_signal = (
        np.sqrt(
            local_sparsity[:, None]
            * local_sparsity[None, :]
        )
        * (1.0 - edge_topness)
    )

    diversity_signal = (
        np.clip(1.0 - overlap, 0.0, 1.0)
        * competitive
    )

    structural_bonus = (
        1.0
        + 0.26 * mutual_nearest
        + 0.10 * mutual_top
    )

    base_scores = (
        distance_score
        * endpoint_score
        * (0.56 + 0.44 * reciprocal_rank)
        * structural_bonus
        * opportunity_penalty
        * scarcity_bonus
        * isolation_bonus
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    log_scores = np.log(base_scores)

    log_scores += (
        0.055
        * diversity_signal
    )

    log_scores += (
        0.045
        * complementary_opportunity
        * competitive
    )

    log_scores += (
        0.035
        * sparse_alternative_signal
    )

    log_scores -= (
        0.065
        * concentration_pressure
    )

    scores = np.exp(
        np.clip(log_scores, -60.0, 30.0)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
