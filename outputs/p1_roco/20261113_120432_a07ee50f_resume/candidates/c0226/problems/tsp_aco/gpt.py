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

    eps = np.finfo(float).eps
    positive = distance[valid & (distance > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1)
    ranks = np.zeros((n, n), dtype=float)
    preferences = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    candidate_counts = np.zeros(n, dtype=float)

    neighbor_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n))))
    )

    for i in range(n):
        candidates = order[i][np.isfinite(distance[i, order[i]])]
        if candidates.size == 0:
            continue

        row_distances = distance[i, candidates]
        candidate_counts[i] = float(candidates.size)

        ranks[i, candidates] = 1.0 / np.arange(
            1.0,
            candidates.size + 1.0,
            dtype=float,
        )

        top_mask[i, candidates[:neighbor_count]] = True
        nearest_mask[i, candidates[0]] = True

        second_distance[i] = (
            row_distances[1]
            if candidates.size > 1
            else row_distances[0]
        )

        positive_neighbors = row_distances[row_distances > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        :min(5, positive_neighbors.size)
                    ]
                )
            )

        row_scale = max(local_scale[i], global_scale * 0.05)
        logits = -row_distances / max(row_scale, eps)
        logits = np.clip(logits, -60.0, 0.0)

        weights = np.exp(logits - np.max(logits))
        total = np.sum(weights)

        if np.isfinite(total) and total > 0.0:
            preferences[i, candidates] = weights / total

    local_scale = np.maximum(
        local_scale,
        global_scale * 0.05,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = distance / np.maximum(pair_scale, eps)
    normalized_distance = np.clip(
        normalized_distance,
        0.0,
        60.0,
    )

    rank_continuity = np.sqrt(ranks * ranks.T)
    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    endpoint_support = np.maximum(
        preferences,
        preferences.T,
    )

    mutual_top = top_mask & top_mask.T
    mutual_nearest = nearest_mask & nearest_mask.T

    scarcity = np.ones((n, n), dtype=float)

    for i in range(n):
        row_mask = valid[i]
        if not np.any(row_mask):
            continue

        reference_gap = max(
            second_distance[i],
            global_scale,
        )
        ratios = reference_gap / np.maximum(
            distance[i, row_mask],
            eps,
        )
        scarcity[i, row_mask] += 0.40 * np.clip(
            ratios - 1.0,
            0.0,
            5.0,
        )

    scarcity = np.sqrt(scarcity * scarcity.T)

    endpoint_temperature = np.sqrt(
        np.maximum(
            global_scale / local_scale,
            0.05,
        )[:, None]
        * np.maximum(
            global_scale / local_scale,
            0.05,
        )[None, :]
    )
    endpoint_temperature = np.clip(
        endpoint_temperature,
        0.35,
        3.0,
    )

    log_score = -1.25 * normalized_distance
    log_score += 0.55 * np.log(
        np.maximum(rank_continuity, eps)
    )
    log_score += 0.85 * np.log(
        np.maximum(mutual_preference, eps)
    )
    log_score += 0.45 * np.log(
        np.maximum(endpoint_support, eps)
    )
    log_score += 0.35 * np.log(
        np.maximum(scarcity, eps)
    )
    log_score /= endpoint_temperature

    log_score += 0.55 * mutual_top.astype(float)
    log_score += 0.35 * mutual_nearest.astype(float)

    log_score[~valid] = -np.inf

    finite_scores = np.isfinite(log_score)
    if not np.any(finite_scores):
        return scores

    maximum = np.max(log_score[finite_scores])
    scores[finite_scores] = np.exp(
        np.clip(
            log_score[finite_scores] - maximum,
            -60.0,
            0.0,
        )
    )

    scores = 0.5 * (scores + scores.T)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores
