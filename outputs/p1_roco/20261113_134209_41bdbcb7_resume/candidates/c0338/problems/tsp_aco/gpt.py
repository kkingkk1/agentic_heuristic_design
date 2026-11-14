import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    forward = np.where(
        np.isfinite(distances) & (distances >= 0.0) & ~diagonal,
        distances,
        np.inf,
    )
    backward = forward.T

    forward_valid = np.isfinite(forward)
    backward_valid = np.isfinite(backward)
    both_valid = forward_valid & backward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        forward[both_valid] + backward[both_valid]
    )

    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid
    distance[only_forward] = forward[only_forward]
    distance[only_backward] = backward[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")

    rank_weight = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    candidate_counts = valid.sum(axis=1).astype(float)
    local_scale = np.full(n, global_scale, dtype=float)
    first_distance = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(
        n - 1,
        max(4, int(np.ceil(1.4 * np.sqrt(n)))),
    )

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        count = candidates.size

        rank_weight[i, candidates] = 1.0 / np.sqrt(
            np.arange(1, count + 1, dtype=float)
        )
        top_mask[i, candidates[:neighborhood_size]] = True
        nearest_mask[i, candidates[0]] = True

        first_distance[i] = values[0]
        second_distance[i] = values[1] if count > 1 else values[0]

        positive_neighbors = values[values > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        :min(7, positive_neighbors.size)
                    ]
                )
            )
        else:
            local_scale[i] = global_scale

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

    relative_distance = np.divide(
        distance,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    endpoint_ratio = np.divide(
        distance,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio = np.clip(endpoint_ratio, 0.0, 50.0)

    reverse_endpoint_ratio = np.divide(
        distance,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    reverse_endpoint_ratio = np.clip(reverse_endpoint_ratio, 0.0, 50.0)

    distance_score = np.exp(-1.08 * relative_distance)
    endpoint_score = np.sqrt(
        np.exp(-endpoint_ratio)
        * np.exp(-reverse_endpoint_ratio)
    )

    logits = (
        -1.0 * relative_distance
        -0.10 * endpoint_ratio
    )
    logits[~valid] = -np.inf

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_logits = logits[i, row_valid]
        row_max = float(np.max(row_logits))
        weights = np.exp(row_logits - row_max)
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    rank_agreement = np.sqrt(
        rank_weight * rank_weight.T
    )

    mutual_nearest = (
        nearest_mask
        & nearest_mask.T
        & valid
    ).astype(float)

    mutual_top = (
        top_mask
        & top_mask.T
        & valid
    ).astype(float)

    second_pair = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    )

    scarcity_ratio = np.divide(
        second_pair,
        distance + 0.5 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )

    scarcity_bonus = 1.0 + 0.18 * np.clip(
        scarcity_ratio - 0.75,
        0.0,
        3.0,
    )

    endpoint_support = np.maximum(
        preferences,
        preferences.T,
    )

    support_factor = 0.72 + 0.28 * endpoint_support

    endpoint_rank_support = np.maximum(
        rank_weight,
        rank_weight.T,
    )

    rank_factor = 0.76 + 0.24 * endpoint_rank_support

    candidate_density = np.sqrt(
        np.maximum(candidate_counts[:, None], 1.0)
        * np.maximum(candidate_counts[None, :], 1.0)
    )
    candidate_density = np.sqrt(
        np.clip(
            candidate_density / max(float(n - 1), 1.0),
            0.45,
            1.0,
        )
    )
    density_factor = 0.88 + 0.12 * candidate_density

    opportunity_ratio = np.divide(
        distance,
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

    shared_neighbors = (
        top_mask.astype(float) @ top_mask.astype(float).T
    )
    shared_neighbor_factor = 1.0 + 0.06 * np.clip(
        shared_neighbors / max(float(neighborhood_size), 1.0),
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_score
        * mutual_preference
        * (0.48 + 0.52 * rank_agreement)
        * support_factor
        * rank_factor
        * scarcity_bonus
        * density_factor
        * opportunity_penalty
        * shared_neighbor_factor
        * (1.0 + 0.52 * mutual_top)
        * (1.0 + 0.34 * mutual_nearest)
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
