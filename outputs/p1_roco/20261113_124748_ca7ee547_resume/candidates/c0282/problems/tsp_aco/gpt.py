import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    raw_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(raw_valid):
        return result

    safe = np.where(raw_valid, distances, np.inf)

    # Use the smallest available directional value to obtain one
    # symmetric effective edge representation.
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    top_neighbors = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, -1, dtype=int)
    valid_counts = np.sum(valid, axis=1).astype(float)

    neighborhood_size = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)
    scale_size = min(5, n - 1)

    for i in range(n):
        neighbors = order[i][valid[i, order[i]]]

        if neighbors.size == 0:
            continue

        values = safe[i, neighbors]
        ranks[i, neighbors] = 1.0 / np.arange(
            1,
            neighbors.size + 1,
            dtype=float,
        )

        nearest[i] = neighbors[0]
        top_neighbors[i, neighbors[:neighborhood_size]] = True

        if neighbors.size > 1:
            second_distance[i] = values[1]
        else:
            second_distance[i] = values[0]

        local_values = values[:scale_size]
        positive_local = local_values[local_values > 0.0]

        if positive_local.size:
            local_scale[i] = float(np.median(positive_local))
        else:
            local_scale[i] = float(np.median(local_values))

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    normalized_distance = safe / np.maximum(pair_scale, eps)
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    distance_score = np.exp(-normalized_distance)

    endpoint_affinity = np.sqrt(
        np.exp(
            -np.clip(
                safe / np.maximum(local_scale[:, None], eps),
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                safe / np.maximum(local_scale[None, :], eps),
                0.0,
                50.0,
            )
        )
    )

    reciprocal_rank = np.sqrt(ranks * ranks.T)
    reciprocal_rank = np.nan_to_num(
        reciprocal_rank,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )

    mutual_local = top_neighbors & top_neighbors.T & valid

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe / np.maximum(endpoint_best, eps)
    opportunity_penalty = 1.0 / (
        1.0
        + 0.16 * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    gap = np.maximum(
        second_distance[:, None],
        second_distance[None, :],
    )

    scarcity = 1.0 + 0.28 * np.clip(
        gap / np.maximum(safe, eps) - 1.0,
        0.0,
        5.0,
    )
    scarcity = np.sqrt(scarcity * scarcity.T)

    # Estimate whether each endpoint has alternatives besides its strongest edge.
    local_strength = np.where(
        valid,
        np.exp(
            -np.clip(
                safe / np.maximum(local_scale[:, None], eps),
                0.0,
                50.0,
            )
        )
        / np.sqrt(1.0 + np.maximum(ranks, 0.0)),
        0.0,
    )

    strongest = np.max(local_strength, axis=1)
    total = np.sum(local_strength, axis=1)

    concentration = strongest / np.maximum(total, eps)
    concentration = np.clip(concentration, 0.0, 1.0)

    sorted_strength = np.sort(local_strength, axis=1)
    second_strength = sorted_strength[:, -2]
    alternative_ratio = second_strength / np.maximum(strongest, eps)
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + ranks), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + ranks.T), 0.0, 1.0)
    )

    sparsity = local_scale / np.maximum(global_scale, eps)
    sparsity = np.clip(sparsity, 0.0, 2.0)
    sparsity = sparsity / (1.0 + sparsity)

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

    scores = (
        distance_score
        * endpoint_affinity
        * (0.54 + 0.46 * reciprocal_rank)
        * (1.0 + 0.34 * mutual_local.astype(float))
        * (1.0 + 0.42 * mutual_nearest.astype(float))
        * opportunity_penalty
        * scarcity
    )

    scores[~valid] = 0.0
    scores = np.maximum(scores, eps)

    log_scores = np.log(scores)
    log_scores += (
        0.06
        * complementary_opportunity
        * (edge_topness > 0.0)
    )
    log_scores += 0.04 * sparse_alternative_bonus
    log_scores -= 0.06 * concentration_penalty

    scores = np.exp(np.clip(log_scores, -60.0, 30.0))
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
