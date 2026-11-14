import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    raw = distance_matrix.copy()
    raw[~np.isfinite(raw)] = np.inf
    raw[raw < 0.0] = 0.0
    np.fill_diagonal(raw, np.inf)

    reverse = raw.T
    forward_valid = np.isfinite(raw)
    reverse_valid = np.isfinite(reverse)

    distance = np.full((n, n), np.inf, dtype=float)
    both = forward_valid & reverse_valid
    only_forward = forward_valid & ~reverse_valid
    only_reverse = reverse_valid & ~forward_valid

    distance[both] = 0.5 * (raw[both] + reverse[both])
    distance[only_forward] = raw[only_forward]
    distance[only_reverse] = reverse[only_reverse]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    positive = distance[valid & (distance > 0.0)]

    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    k = min(
        n - 1,
        max(4, int(np.ceil(1.5 * np.sqrt(n))))
    )

    order = np.argsort(distance, axis=1)
    ranks = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    first_distance = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    degree = np.zeros(n, dtype=float)

    for i in range(n):
        candidates = order[i][np.isfinite(distance[i, order[i]])]
        if candidates.size == 0:
            continue

        row_distances = distance[i, candidates]
        degree[i] = candidates.size

        ranks[i, candidates] = 1.0 / np.sqrt(
            np.arange(1, candidates.size + 1, dtype=float)
        )
        top_mask[i, candidates[:k]] = True
        nearest_mask[i, candidates[0]] = True

        first_distance[i] = row_distances[0]
        second_distance[i] = (
            row_distances[1]
            if candidates.size > 1
            else row_distances[0]
        )

        positive_row = row_distances[row_distances > 0.0]
        if positive_row.size:
            local_scale[i] = float(
                np.median(positive_row[:min(8, positive_row.size)])
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(
        local_scale,
        max(global_scale * 0.025, eps)
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = distance / np.maximum(pair_scale, eps)
    normalized_distance = np.clip(normalized_distance, 0.0, 80.0)

    scale_ratio = np.sqrt(
        np.maximum(global_scale / local_scale[:, None], 0.05)
        * np.maximum(global_scale / local_scale[None, :], 0.05)
    )
    beta = np.clip(scale_ratio, 0.35, 2.75)

    logits = -normalized_distance * beta
    logits[~valid] = -np.inf

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_logits = logits[i, row_valid]
        offset = np.max(row_logits)
        weights = np.exp(row_logits - offset)
        total = np.sum(weights)

        if np.isfinite(total) and total > 0.0:
            preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    rank_continuity = np.sqrt(
        ranks * ranks.T
    )

    mutual_top = top_mask & top_mask.T
    mutual_nearest = nearest_mask & nearest_mask.T

    endpoint_gap = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        denominator = max(
            second_distance[i],
            local_scale[i],
            eps
        )
        improvement = (
            second_distance[i] - distance[i, row_valid]
        ) / denominator

        endpoint_gap[i, row_valid] = 1.0 + 0.85 * np.clip(
            improvement,
            0.0,
            1.5
        )

    gap_support = np.sqrt(
        endpoint_gap * endpoint_gap.T
    )

    endpoint_support = np.maximum(
        preferences,
        preferences.T
    )

    degree_ratio = np.sqrt(
        np.maximum(degree[:, None], 1.0)
        * np.maximum(degree[None, :], 1.0)
    )
    density_support = np.clip(
        np.sqrt(
            np.maximum(n - 1, 1.0) / degree_ratio
        ),
        0.72,
        1.15
    )

    local_scarcity = 1.0 / (
        1.0
        + normalized_distance
        / np.maximum(
            np.sqrt(
                np.maximum(
                    first_distance[:, None],
                    eps
                )
                * np.maximum(
                    first_distance[None, :],
                    eps
                )
            ),
            eps
        )
    )
    local_scarcity = 0.72 + 0.58 * local_scarcity

    exploration_floor = np.zeros((n, n), dtype=float)
    exploration_floor[valid] = 0.08

    scores = (
        0.58 * mutual_preference
        + 0.42 * rank_continuity
    )
    scores *= 0.70 + 0.30 * endpoint_support
    scores *= gap_support
    scores *= local_scarcity
    scores *= density_support
    scores *= 1.0 + 0.72 * mutual_top.astype(float)
    scores *= 1.0 + 0.62 * mutual_nearest.astype(float)
    scores += exploration_floor * np.exp(
        -0.55 * normalized_distance
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores
