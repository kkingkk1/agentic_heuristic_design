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
    finite_raw = np.isfinite(raw)

    raw[~finite_raw] = np.inf
    raw[raw < 0] = 0.0
    np.fill_diagonal(raw, np.inf)

    transpose = raw.T
    finite = np.isfinite(raw)
    finite_t = np.isfinite(transpose)

    both = finite & finite_t
    forward_only = finite & ~finite_t
    backward_only = finite_t & ~finite

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both] = 0.5 * (raw[both] + transpose[both])
    distance[forward_only] = raw[forward_only]
    distance[backward_only] = transpose[backward_only]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    positive = distance[valid & (distance > 0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    neighbor_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n))))
    )

    order = np.argsort(distance, axis=1)
    ranks = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    for i in range(n):
        candidates = order[i][np.isfinite(distance[i, order[i]])]

        if candidates.size == 0:
            continue

        row_distances = distance[i, candidates]

        rank_values = np.arange(1, candidates.size + 1, dtype=float)
        ranks[i, candidates] = 1.0 / rank_values

        top = candidates[:neighbor_count]
        top_mask[i, top] = True
        nearest_mask[i, candidates[0]] = True

        if candidates.size > 1:
            second_distance[i] = row_distances[1]
        else:
            second_distance[i] = row_distances[0]

        positive_neighbors = row_distances[row_distances > 0]
        if positive_neighbors.size:
            count = min(5, positive_neighbors.size)
            local_scale[i] = float(
                np.median(positive_neighbors[:count])
            )

    local_scale = np.maximum(local_scale, global_scale * 0.05)

    normalized_distance = distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    logits = -np.clip(normalized_distance, 0.0, 60.0)
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

        if total > 0 and np.isfinite(total):
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
        ratio = gap / np.maximum(
            distance[i, row_mask],
            np.finfo(float).eps
        )

        scarcity[i, row_mask] += 0.30 * np.clip(
            ratio - 1.0,
            0.0,
            4.0
        )

    scarcity = np.sqrt(scarcity * scarcity.T)

    endpoint_importance = np.maximum(
        preferences,
        preferences.T
    )

    redundancy_penalty = 1.0 - 0.20 * (
        1.0 - rank_continuity
    )

    scores = (
        mutual_preference
        * (0.55 + 0.45 * rank_continuity)
        * (1.0 + 0.65 * mutual_top.astype(float))
        * (1.0 + 0.35 * mutual_nearest.astype(float))
        * scarcity
        * (0.75 + 0.25 * endpoint_importance)
        * redundancy_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores
