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

    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return scores

    raw = np.where(raw_valid, distances, np.inf)
    reverse = raw.T

    forward_valid = np.isfinite(raw)
    reverse_valid = np.isfinite(reverse)
    both_valid = forward_valid & reverse_valid

    safe = np.full((n, n), np.inf, dtype=float)
    safe[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )
    safe[forward_valid & ~reverse_valid] = raw[
        forward_valid & ~reverse_valid
    ]
    safe[reverse_valid & ~forward_valid] = reverse[
        reverse_valid & ~forward_valid
    ]

    np.fill_diagonal(safe, np.inf)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return scores

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    rank = np.full((n, n), float(n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    np.put_along_axis(
        rank,
        order,
        np.broadcast_to(rank_values, (n, n)),
        axis=1,
    )
    rank[~valid] = float(n)

    neighbor_count = min(
        n - 1,
        max(3, int(np.ceil(1.35 * np.sqrt(n)))),
    )

    top_neighbors = np.zeros((n, n), dtype=bool)
    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    valid_counts = valid.sum(axis=1).astype(float)

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        values = safe[i, candidates]
        top_neighbors[i, candidates[:neighbor_count]] = True

        positive = values[values > 0.0]
        if positive.size:
            local_scale[i] = float(
                np.median(positive[: min(7, positive.size)])
            )

        second_distance[i] = (
            values[1] if values.size > 1 else values[0]
        )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    safe_for_math = np.where(valid, safe, np.inf)

    normalized_distance = safe_for_math / np.maximum(
        pair_scale,
        eps,
    )
    normalized_distance = np.clip(
        normalized_distance,
        0.0,
        50.0,
    )

    distance_kernel = np.exp(-1.10 * normalized_distance)

    endpoint_kernel = np.sqrt(
        np.exp(
            -np.clip(
                safe_for_math / local_scale[:, None],
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                safe_for_math / local_scale[None, :],
                0.0,
                50.0,
            )
        )
    )

    rank_kernel = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    top_neighbors_float = top_neighbors.astype(float)
    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = order[i][valid[i, order[i]]]
        if candidates.size:
            nearest[i] = candidates[0]

    indices = np.arange(n)
    mutual_nearest = (
        valid
        & (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    ).astype(float)

    competitive_k = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n)))),
    )
    competitive = (
        valid
        & (rank < float(competitive_k))
        & (rank.T < float(competitive_k))
    ).astype(float)

    shared_neighbors = (
        top_neighbors_float @ top_neighbors_float.T
    )

    union_neighbors = (
        top_neighbors_float.sum(axis=1)[:, None]
        + top_neighbors_float.sum(axis=1)[None, :]
        - shared_neighbors
    )

    neighborhood_overlap = shared_neighbors / np.maximum(
        union_neighbors,
        1.0,
    )

    complementary_bonus = (
        1.0
        + 0.15
        * competitive
        * np.clip(1.0 - neighborhood_overlap, 0.0, 1.0)
    )

    two_hop_bonus = (
        1.0
        + 0.06
        * competitive
        * np.clip(
            shared_neighbors / max(float(neighbor_count), 1.0),
            0.0,
            1.0,
        )
    )

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe_for_math / np.maximum(
        endpoint_best,
        eps,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.12 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    scarcity_ratio = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    ) / np.maximum(safe_for_math, eps)

    scarcity_bonus = 1.0 + 0.10 * np.clip(
        scarcity_ratio - 1.0,
        0.0,
        4.0,
    )

    candidate_popularity = top_neighbors_float.sum(axis=0)
    popularity_reference = max(
        float(np.mean(candidate_popularity)),
        1.0,
    )
    popularity_ratio = np.clip(
        candidate_popularity / popularity_reference,
        0.0,
        3.0,
    )

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + 0.14 * popularity_ratio[:, None])
        * (1.0 + 0.14 * popularity_ratio[None, :])
    )

    isolation_bonus = 1.0 + 0.10 * np.clip(
        pair_scale / np.maximum(safe_for_math + global_scale, eps),
        0.0,
        1.0,
    )

    valid_count_scale = np.sqrt(
        np.maximum(valid_counts[:, None], 1.0)
        * np.maximum(valid_counts[None, :], 1.0)
    )
    density_factor = np.clip(
        np.sqrt(valid_count_scale / max(float(n - 1), 1.0)),
        0.82,
        1.0,
    )

    mutual_bonus = (
        1.0
        + 0.25 * mutual_nearest
        + 0.08 * mutual_top
    )

    scores = (
        distance_kernel
        * endpoint_kernel
        * (0.56 + 0.44 * rank_kernel)
        * mutual_bonus
        * opportunity_penalty
        * scarcity_bonus
        * two_hop_bonus
        * complementary_bonus
        * concentration_penalty
        * isolation_bonus
        * density_factor
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
