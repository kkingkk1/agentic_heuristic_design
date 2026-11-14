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

    eps = np.finfo(float).eps

    raw = distance_matrix.copy()
    raw[~np.isfinite(raw)] = np.inf
    raw[raw < 0.0] = 0.0
    np.fill_diagonal(raw, np.inf)

    reverse = raw.T
    forward_valid = np.isfinite(raw)
    reverse_valid = np.isfinite(reverse)

    both_valid = forward_valid & reverse_valid
    forward_only = forward_valid & ~reverse_valid
    reverse_only = reverse_valid & ~forward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )
    distance[forward_only] = raw[forward_only]
    distance[reverse_only] = reverse[reverse_only]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    positive = distance[valid & (distance > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    neighbor_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n))))
    )

    order = np.argsort(distance, axis=1)
    ranks = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)
    second_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    candidate_counts = np.zeros(n, dtype=float)

    for i in range(n):
        candidates = order[i][np.isfinite(distance[i, order[i]])]
        if candidates.size == 0:
            continue

        row_distances = distance[i, candidates]
        candidate_counts[i] = float(candidates.size)

        ranks[i, candidates] = 1.0 / np.arange(
            1,
            candidates.size + 1,
            dtype=float,
        )

        top_mask[i, candidates[:neighbor_count]] = True
        nearest_mask[i, candidates[0]] = True

        if candidates.size > 1:
            second_mask[i, candidates[1]] = True
            second_distance[i] = row_distances[1]
        else:
            second_distance[i] = row_distances[0]

        positive_neighbors = row_distances[row_distances > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        :min(5, positive_neighbors.size)
                    ]
                )
            )

    local_scale = np.maximum(
        local_scale,
        0.05 * global_scale,
    )

    scale_ratio = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = distance / np.maximum(scale_ratio, eps)
    normalized_distance = np.clip(
        normalized_distance,
        0.0,
        50.0,
    )

    endpoint_temperature = np.sqrt(
        np.maximum(global_scale / local_scale, 0.05)[:, None]
        * np.maximum(global_scale / local_scale, 0.05)[None, :]
    )
    endpoint_temperature = np.clip(
        endpoint_temperature,
        0.35,
        2.5,
    )

    logits = -normalized_distance * endpoint_temperature
    logits[~valid] = -np.inf

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_logits = logits[i, row_valid]
        row_max = np.max(row_logits)
        weights = np.exp(row_logits - row_max)
        total = np.sum(weights)

        if np.isfinite(total) and total > 0.0:
            preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )
    endpoint_preference = np.maximum(
        preferences,
        preferences.T,
    )

    rank_continuity = np.sqrt(
        ranks * ranks.T
    )
    endpoint_rank_support = np.maximum(
        ranks,
        ranks.T,
    )

    gap_forward = np.ones((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        gap = max(
            second_distance[i] - global_scale * 0.05,
            0.0,
        )
        relative_gap = gap / np.maximum(
            local_scale[i],
            eps,
        )

        gap_forward[i, row_valid] += 0.30 * np.clip(
            relative_gap,
            0.0,
            4.0,
        )

    gap_factor = np.sqrt(
        gap_forward * gap_forward.T
    )

    mutual_top = top_mask & top_mask.T
    mutual_nearest = nearest_mask & nearest_mask.T
    shared_second = second_mask & second_mask.T

    density = np.sqrt(
        np.clip(candidate_counts[:, None], 1.0, None)
        * np.clip(candidate_counts[None, :], 1.0, None)
    )
    density_factor = np.sqrt(
        density / max(float(n - 1), 1.0)
    )
    density_factor = np.clip(
        density_factor,
        0.70,
        1.0,
    )

    local_closeness = np.exp(
        -np.minimum(normalized_distance, 30.0)
    )
    local_closeness[~valid] = 0.0

    support = (
        0.45 * mutual_preference
        + 0.55 * endpoint_preference
    )

    rank_factor = (
        0.38
        + 0.62 * rank_continuity
    )

    redundancy_factor = (
        0.70
        + 0.30 * endpoint_rank_support
    )

    structural_bonus = (
        1.0
        + 0.42 * mutual_top.astype(float)
        + 0.20 * mutual_nearest.astype(float)
        + 0.12 * shared_second.astype(float)
    )

    scores = (
        support
        * rank_factor
        * (0.58 + 0.42 * local_closeness)
        * redundancy_factor
        * gap_factor
        * density_factor
        * structural_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores
