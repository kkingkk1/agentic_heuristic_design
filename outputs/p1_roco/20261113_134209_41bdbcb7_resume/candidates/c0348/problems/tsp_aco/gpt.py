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

    forward = np.where(finite, distances, np.inf)
    backward = forward.T

    both_valid = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = np.isfinite(backward) & ~np.isfinite(forward)

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        forward[both_valid] + backward[both_valid]
    )
    distance[only_forward] = forward[only_forward]
    distance[only_backward] = backward[only_backward]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
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
    ranks = np.zeros((n, n), dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    candidate_counts = valid.sum(axis=1).astype(float)
    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.5 * np.sqrt(n)))),
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

        positive = values[values > 0.0]
        if positive.size:
            local_scale[i] = float(
                np.median(positive[: min(6, positive.size)])
            )

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

    normalized_distance = np.divide(
        distance,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    endpoint_distance = np.divide(
        distance,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_distance = np.clip(endpoint_distance, 0.0, 50.0)

    reverse_endpoint_distance = np.divide(
        distance,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    reverse_endpoint_distance = np.clip(
        reverse_endpoint_distance,
        0.0,
        50.0,
    )

    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_scale = max(local_scale[i], 0.15 * global_scale, eps)
        logits = -(
            0.90 * distance[i, row_valid] / row_scale
            + 0.16 * endpoint_distance[i, row_valid]
        )

        maximum = float(np.max(logits))
        weights = np.exp(np.clip(logits - maximum, -60.0, 0.0))
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        np.maximum(preferences * preferences.T, 0.0)
    )

    rank_agreement = np.sqrt(
        np.maximum(ranks * ranks.T, 0.0)
    )

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

    top_float = top_mask.astype(float)
    shared_neighbors = top_float @ top_float.T
    shared_neighbor_signal = np.divide(
        shared_neighbors,
        np.maximum(
            top_mask.sum(axis=1)[:, None]
            + top_mask.sum(axis=1)[None, :]
            - shared_neighbors,
            1.0,
        ),
    )
    shared_neighbor_signal = np.clip(
        shared_neighbor_signal,
        0.0,
        1.0,
    )

    second_pair = np.sqrt(
        np.maximum(second_distance[:, None], eps)
        * np.maximum(second_distance[None, :], eps)
    )

    regret_ratio = np.divide(
        second_pair,
        distance + 0.20 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    regret_bonus = np.clip(regret_ratio, 0.0, 5.0)

    density = np.sqrt(
        np.clip(
            candidate_counts[:, None]
            * candidate_counts[None, :]
            / max(float(n - 1) ** 2, 1.0),
            0.25,
            1.0,
        )
    )

    log_scores = np.zeros((n, n), dtype=float)

    log_scores += -1.20 * normalized_distance
    log_scores += -0.22 * (
        endpoint_distance + reverse_endpoint_distance
    )

    log_scores += 0.62 * np.log(
        np.maximum(mutual_preference, eps)
    )
    log_scores += 0.42 * np.log(
        0.30 + 0.70 * rank_agreement
    )

    log_scores += 0.34 * np.log1p(
        1.8 * np.maximum(preferences, preferences.T)
    )
    log_scores += 0.48 * mutual_top
    log_scores += 0.26 * mutual_nearest
    log_scores += 0.10 * shared_neighbor_signal
    log_scores += 0.16 * np.minimum(regret_bonus, 2.5)
    log_scores += np.log(0.90 + 0.10 * density)

    log_scores[~valid] = -np.inf

    finite_log_scores = log_scores[np.isfinite(log_scores)]
    if finite_log_scores.size == 0:
        return scores

    maximum = float(np.max(finite_log_scores))
    scores = np.zeros((n, n), dtype=float)
    scores[valid] = np.exp(
        np.clip(log_scores[valid] - maximum, -60.0, 0.0)
    )

    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
