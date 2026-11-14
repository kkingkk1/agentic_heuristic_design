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
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal
    raw = np.where(finite, distances, np.inf)

    reverse = raw.T
    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)
    both_valid = forward_valid & backward_valid

    distance = np.full((n, n), np.inf, dtype=float)
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
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    ranks = np.zeros((n, n), dtype=float)
    preferences = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.25 * np.sqrt(n)))),
    )

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        ranks[i, candidates] = 1.0 / (
            1.0 + np.arange(candidates.size, dtype=float)
        )

        top_mask[i, candidates[:neighborhood_size]] = True
        nearest_mask[i, candidates[0]] = True

        if candidates.size > 1:
            second_distance[i] = values[1]
        else:
            second_distance[i] = values[0]

        positive_neighbors = values[values > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        : min(6, positive_neighbors.size)
                    ]
                )
            )

        scale = max(local_scale[i], 0.05 * global_scale, eps)
        logits = -distance[i, candidates] / scale
        logits -= 0.20 * np.arange(candidates.size, dtype=float)

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

    global_relative = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    global_relative = np.clip(global_relative, 0.0, 50.0)

    endpoint_ratio = np.divide(
        distance,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio_reverse = np.divide(
        distance,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    endpoint_ratio = np.clip(endpoint_ratio, 0.0, 50.0)
    endpoint_ratio_reverse = np.clip(endpoint_ratio_reverse, 0.0, 50.0)

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )
    rank_agreement = np.sqrt(
        ranks * ranks.T
    )

    mutual_top = (
        top_mask & top_mask.T & valid
    ).astype(float)

    mutual_nearest = (
        nearest_mask & nearest_mask.T & valid
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
    scarcity_signal = np.clip(scarcity_ratio - 0.75, 0.0, 2.5)

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
    opportunity_signal = np.clip(opportunity_ratio - 1.0, 0.0, 4.0)

    valid_mutual_preference = np.maximum(mutual_preference, eps)
    valid_rank_agreement = np.maximum(rank_agreement, eps)

    log_scores = np.zeros((n, n), dtype=float)
    log_scores[valid] = (
        -1.22 * relative_distance[valid]
        -0.18 * global_relative[valid]
        +0.28 * np.log(valid_mutual_preference[valid])
        +0.16 * np.log(0.55 + 0.45 * valid_rank_agreement[valid])
        +0.24 * mutual_top[valid]
        +0.12 * mutual_nearest[valid]
        +0.07 * scarcity_signal[valid]
        -0.055 * opportunity_signal[valid]
    )

    maximum = float(np.max(log_scores[valid]))
    weights = np.zeros((n, n), dtype=float)
    weights[valid] = np.exp(
        np.clip(log_scores[valid] - maximum, -60.0, 0.0)
    )

    scores = 0.5 * (weights + weights.T)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
