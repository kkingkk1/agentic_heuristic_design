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

    diagonal = np.eye(n, dtype=bool)

    raw = np.where(
        np.isfinite(distances) & (distances >= 0.0) & ~diagonal,
        distances,
        np.inf,
    )

    reverse = raw.T
    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)
    valid = forward_valid | backward_valid

    distance = np.full((n, n), np.inf, dtype=float)

    both_valid = forward_valid & backward_valid
    distance[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )

    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid
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

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    local_scale = np.full(n, global_scale, dtype=float)
    ranks = np.zeros((n, n), dtype=float)
    preferences = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.25 * np.sqrt(n)))),
    )

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        values = distance[i, candidates]

        rank_values = 1.0 / np.sqrt(
            1.0 + np.arange(candidates.size, dtype=float)
        )
        ranks[i, candidates] = rank_values

        top_candidates = candidates[:neighborhood_size]
        top_mask[i, top_candidates] = True
        nearest_mask[i, candidates[0]] = True

        positive_neighbors = values[values > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[: min(6, positive_neighbors.size)]
                )
            )
        else:
            local_scale[i] = max(float(np.median(values)), global_scale)

        logits = -1.15 * values / max(local_scale[i], eps)
        logits -= 0.10 * np.minimum(
            values / max(local_scale[i], eps),
            8.0,
        )

        maximum = float(np.max(logits))
        weights = np.exp(logits - maximum)
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            preferences[i, candidates] = weights / total

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
        pair_scale,
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

    base_quality = np.exp(
        -0.92 * relative_distance
        -0.08 * endpoint_ratio
    )

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )
    mutual_rank = np.sqrt(ranks * ranks.T)

    mutual_top = (
        top_mask
        & top_mask.T
        & valid
    ).astype(float)

    mutual_nearest = (
        nearest_mask
        & nearest_mask.T
        & valid
    ).astype(float)

    rank_signal = np.sqrt(
        np.maximum(ranks, 0.0) * np.maximum(ranks.T, 0.0)
    )
    rank_signal = np.divide(
        rank_signal,
        np.maximum(np.max(rank_signal), eps),
    )

    preference_signal = np.sqrt(
        np.maximum(mutual_preference, 0.0)
    )

    core = (
        base_quality
        * (0.72 + 0.28 * preference_signal)
        * (0.84 + 0.16 * mutual_rank)
        * (1.0 + 0.24 * mutual_top)
        * (1.0 + 0.14 * mutual_nearest)
    )

    scores = 0.92 * core + 0.08 * rank_signal
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
