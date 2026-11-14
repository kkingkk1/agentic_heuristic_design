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

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    raw = np.where(
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal,
        distances,
        np.inf,
    )

    reverse = raw.T
    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)

    both_valid = forward_valid & backward_valid
    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid

    safe = np.full((n, n), np.inf, dtype=float)
    safe[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )
    safe[only_forward] = raw[only_forward]
    safe[only_backward] = reverse[only_backward]

    np.fill_diagonal(safe, np.inf)
    valid = np.isfinite(safe)

    if not np.any(valid):
        return result

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
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    valid_counts = valid.sum(axis=1).astype(float)
    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(n - 1, max(3, int(np.ceil(1.4 * np.sqrt(n))))
    )
    top_k = min(n - 1, max(2, int(np.ceil(0.75 * neighborhood_size))))

    for i in range(n):
        neighbors = order[i, valid[i, order[i]]]

        if neighbors.size == 0:
            continue

        rank[i, neighbors] = np.arange(neighbors.size, dtype=float)
        top_mask[i, neighbors[:neighborhood_size]] = True
        nearest_mask[i, neighbors[0]] = True

        values = safe[i, neighbors]
        if neighbors.size >= 2:
            second_distance[i] = values[1]
        else:
            second_distance[i] = values[0]

        local_values = values[: min(6, values.size)]
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
    local_scale = np.maximum(local_scale, 0.05 * global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        safe,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    row_ratio = np.divide(
        safe,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        safe,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    row_ratio = np.clip(row_ratio, 0.0, 50.0)
    col_ratio = np.clip(col_ratio, 0.0, 50.0)

    shortness = np.exp(-1.15 * relative_distance)
    endpoint_affinity = np.exp(
        -0.5 * (row_ratio + col_ratio)
    )

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        logits = (
            -0.92 * relative_distance[i, row_valid]
            -0.10 * row_ratio[i, row_valid]
            -0.08 * rank[i, row_valid]
        )

        maximum = float(np.max(logits))
        weights = np.exp(np.clip(logits - maximum, -60.0, 0.0))
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    rank_agreement = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )
    rank_agreement[~valid] = 0.0

    top_neighbors = np.zeros((n, n), dtype=bool)
    for i in range(n):
        neighbors = order[i, valid[i, order[i]]]
        top_neighbors[i, neighbors[:top_k]] = True

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    mutual_nearest = (
        nearest_mask
        & nearest_mask.T
        & valid
    ).astype(float)

    local_competition = (
        valid
        & (rank < float(top_k))
        & (rank.T < float(top_k))
    ).astype(float)

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    union_size = (
        top_neighbors.sum(axis=1)[:, None]
        + top_neighbors.sum(axis=1)[None, :]
        - shared_neighbors
    )

    neighborhood_overlap = shared_neighbors / np.maximum(
        union_size,
        1.0,
    )

    diversity_bonus = 1.0 + 0.14 * np.clip(
        1.0 - neighborhood_overlap,
        0.0,
        1.0,
    ) * local_competition

    second_pair = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    )

    scarcity_ratio = np.divide(
        second_pair,
        safe + 0.5 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )

    scarcity_bonus = 1.0 + 0.16 * np.clip(
        scarcity_ratio - 0.65,
        0.0,
        2.5,
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

    density_factor = np.sqrt(
        np.clip(
            valid_counts[:, None]
            * valid_counts[None, :]
            / max(float(n - 1) ** 2, 1.0),
            0.35,
            1.0,
        )
    )

    density_adjustment = 0.90 + 0.10 * density_factor

    scores = (
        shortness
        * endpoint_affinity
        * (0.54 + 0.46 * rank_agreement)
        * (0.72 + 0.28 * mutual_preference)
        * (1.0 + 0.28 * mutual_top)
        * (1.0 + 0.18 * mutual_nearest)
        * (1.0 + 0.08 * local_competition)
        * diversity_bonus
        * scarcity_bonus
        * opportunity_penalty
        * density_adjustment
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
