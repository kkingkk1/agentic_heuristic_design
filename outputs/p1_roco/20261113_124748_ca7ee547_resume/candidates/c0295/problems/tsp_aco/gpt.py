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
    raw[raw < 0] = 0.0
    np.fill_diagonal(raw, np.inf)

    transposed = raw.T
    finite_forward = np.isfinite(raw)
    finite_backward = np.isfinite(transposed)

    both = finite_forward & finite_backward
    only_forward = finite_forward & ~finite_backward
    only_backward = finite_backward & ~finite_forward

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both] = 0.5 * (raw[both] + transposed[both])
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = transposed[only_backward]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    positive = distance[valid & (distance > 0)]
    eps = np.finfo(float).eps

    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    neighbor_count = min(
        n - 1,
        max(3, int(np.ceil(1.35 * np.sqrt(n))))
    )

    order = np.argsort(distance, axis=1)
    ranks = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    candidate_counts = np.zeros(n, dtype=float)

    for i in range(n):
        candidates = order[i][np.isfinite(distance[i, order[i]])]
        if candidates.size == 0:
            continue

        row_distances = distance[i, candidates]
        candidate_counts[i] = candidates.size

        ranks[i, candidates] = 1.0 / np.arange(
            1,
            candidates.size + 1,
            dtype=float,
        )

        top_mask[i, candidates[:neighbor_count]] = True
        nearest_mask[i, candidates[0]] = True

        second_distance[i] = (
            row_distances[1]
            if candidates.size > 1
            else row_distances[0]
        )

        positive_neighbors = row_distances[row_distances > 0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        :min(7, positive_neighbors.size)
                    ]
                )
            )

    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = distance / np.maximum(pair_scale, eps)
    normalized_distance = np.clip(normalized_distance, 0.0, 60.0)

    scale_ratio = np.maximum(
        global_scale / local_scale,
        0.05,
    )
    endpoint_beta = np.sqrt(
        scale_ratio[:, None] * scale_ratio[None, :]
    )

    logits = -normalized_distance * np.clip(
        endpoint_beta,
        0.28,
        3.10,
    )
    logits[~valid] = -np.inf

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_mask = valid[i]
        if not np.any(row_mask):
            continue

        row_logits = logits[i, row_mask]
        maximum = np.max(row_logits)
        weights = np.exp(row_logits - maximum)
        total = np.sum(weights)

        if np.isfinite(total) and total > 0:
            preferences[i, row_mask] = weights / total

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    rank_continuity = np.sqrt(
        ranks * ranks.T
    )

    mutual_top = top_mask & top_mask.T
    mutual_nearest = nearest_mask & nearest_mask.T

    scarcity = np.ones((n, n), dtype=float)

    for i in range(n):
        row_mask = valid[i]
        if not np.any(row_mask):
            continue

        gap = max(second_distance[i], global_scale)
        ratio = gap / np.maximum(distance[i, row_mask], eps)

        scarcity[i, row_mask] += 0.50 * np.clip(
            ratio - 1.0,
            0.0,
            5.0,
        )

    scarcity = np.sqrt(scarcity * scarcity.T)

    endpoint_support = np.maximum(
        preferences,
        preferences.T,
    )

    endpoint_rank_support = np.maximum(
        ranks,
        ranks.T,
    )

    redundancy_penalty = (
        0.72
        + 0.28 * endpoint_rank_support
    )

    density_factor = np.sqrt(
        np.clip(candidate_counts[:, None], 1.0, None)
        * np.clip(candidate_counts[None, :], 1.0, None)
    )
    density_factor = np.clip(
        np.sqrt(density_factor / max(n - 1, 1)),
        0.78,
        1.0,
    )

    scores = (
        mutual_preference
        * (0.43 + 0.57 * rank_continuity)
        * (0.73 + 0.27 * endpoint_support)
        * redundancy_penalty
        * scarcity
        * density_factor
        * (1.0 + 0.86 * mutual_top.astype(float))
        * (1.0 + 0.54 * mutual_nearest.astype(float))
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores
