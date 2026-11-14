import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
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
    backward_valid = np.isfinite(reverse)

    both_valid = forward_valid & backward_valid
    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = reverse[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    rank = np.full((n, n), float(n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest = np.full(n, -1, dtype=int)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    valid_counts = valid.sum(axis=1).astype(float)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.4 * np.sqrt(n)))),
    )

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]
        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        rank[i, candidates] = np.arange(
            candidates.size,
            dtype=float,
        )
        top_mask[i, candidates[:neighborhood_size]] = True
        nearest[i] = int(candidates[0])

        if candidates.size > 1:
            second_distance[i] = float(values[1])
        else:
            second_distance[i] = float(values[0])

        positive_neighbors = values[values > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        :min(6, positive_neighbors.size)
                    ]
                )
            )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(
        local_scale,
        0.05 * global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = np.divide(
        distance,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    endpoint_ratio = np.divide(
        distance,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio = np.clip(endpoint_ratio, 0.0, 50.0)

    endpoint_ratio_t = endpoint_ratio.T

    distance_kernel = np.exp(
        -1.16 * np.power(normalized_distance, 1.08)
    )
    endpoint_kernel = np.exp(
        -0.5 * (endpoint_ratio + endpoint_ratio_t)
    )

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        logits = (
            -1.05 * normalized_distance[i, row_valid]
            -0.16 * endpoint_ratio[i, row_valid]
        )
        maximum = float(np.max(logits))
        weights = np.exp(logits - maximum)
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    rank_agreement = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    mutual_top = (
        top_mask
        & top_mask.T
        & valid
    ).astype(float)

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    shared_neighbors = (
        top_mask.astype(float) @ top_mask.astype(float).T
    )
    shared_neighbor_signal = np.clip(
        shared_neighbors / max(float(neighborhood_size), 1.0),
        0.0,
        1.0,
    )

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
    scarcity_bonus = 1.0 + 0.16 * np.clip(
        scarcity_ratio - 0.8,
        0.0,
        3.0,
    )

    support = np.maximum(preferences, preferences.T)

    local_strength = np.where(
        valid,
        np.exp(-endpoint_ratio) / np.sqrt(1.0 + rank),
        0.0,
    )
    strongest = np.max(local_strength, axis=1)
    sorted_strength = np.sort(local_strength, axis=1)
    second_strength = sorted_strength[:, -2]

    alternative_ratio = np.divide(
        second_strength,
        np.maximum(strongest, eps),
    )
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)

    complement_bonus = 1.0 + 0.08 * np.sqrt(
        alternative_ratio[:, None]
        * alternative_ratio[None, :]
    )

    density_adjustment = 0.92 + 0.08 * np.sqrt(
        np.clip(
            valid_counts[:, None]
            * valid_counts[None, :]
            / max(float(n - 1) ** 2, 1.0),
            0.0,
            1.0,
        )
    )

    scores = (
        distance_kernel
        * endpoint_kernel
        * (0.52 + 0.48 * mutual_preference)
        * (0.55 + 0.45 * rank_agreement)
        * (0.82 + 0.18 * support)
        * (1.0 + 0.34 * mutual_top)
        * (1.0 + 0.24 * mutual_nearest)
        * (1.0 + 0.10 * shared_neighbor_signal)
        * scarcity_bonus
        * complement_bonus
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
