import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, locally adaptive TSP edge desirability priors."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(raw)
    nonnegative = np.maximum(raw, 0.0)
    np.fill_diagonal(finite, False)

    reverse_finite = finite.T
    both_valid = finite & reverse_finite
    forward_only = finite & ~reverse_finite
    backward_only = reverse_finite & ~finite

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        nonnegative[both_valid] + nonnegative.T[both_valid]
    )
    distance[forward_only] = nonnegative[forward_only]
    distance[backward_only] = nonnegative.T[backward_only]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    positive = distance[valid & (distance > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = distance[valid]
        global_scale = (
            float(np.median(finite_values))
            if finite_values.size
            else 1.0
        )

    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)

    local_scale = np.full(n, global_scale, dtype=float)
    nearest_distance = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    nearest_index = np.full(n, -1, dtype=int)

    ranks = np.zeros((n, n), dtype=float)
    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = order[i][np.isfinite(sorted_distance[i])]
        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        nearest_index[i] = candidates[0]
        nearest_distance[i] = values[0]
        second_distance[i] = values[1] if values.size > 1 else values[0]

        rank_values = 1.0 / np.arange(1, candidates.size + 1, dtype=float)
        ranks[i, candidates] = rank_values

        positive_values = values[values > 0.0]
        if positive_values.size:
            local_scale[i] = float(
                np.median(positive_values[: min(5, positive_values.size)])
            )
        elif values.size:
            local_scale[i] = global_scale

        scale = max(local_scale[i], global_scale * 0.05, eps)
        logits = -np.clip(values / scale, 0.0, 60.0)
        weights = np.exp(logits - np.max(logits))
        total = float(weights.sum())

        if np.isfinite(total) and total > 0.0:
            preferences[i, candidates] = weights / total

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized_distance = distance / pair_scale
        shortness = np.exp(-np.clip(normalized_distance, 0.0, 60.0))

    rank_consensus = np.sqrt(ranks * ranks.T)
    mutual_preference = np.sqrt(preferences * preferences.T)

    indices = np.arange(n)
    mutual_nearest = (
        (nearest_index[:, None] == indices[None, :])
        & (nearest_index[None, :] == indices[:, None])
        & (nearest_index[:, None] >= 0)
    )

    gap = np.maximum(second_distance, global_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        scarcity = 1.0 + 0.25 * np.clip(
            gap[:, None] / np.maximum(distance, eps) - 1.0,
            0.0,
            3.0,
        )

    scarcity = np.sqrt(scarcity * scarcity.T)

    endpoint_importance = np.maximum(preferences, preferences.T)
    degree_balance = 0.75 + 0.25 * endpoint_importance
    rank_factor = 0.55 + 0.45 * rank_consensus
    preference_factor = 0.60 + 0.40 * mutual_preference
    neighborhood_bonus = 1.0 + 0.45 * mutual_nearest.astype(float)

    scores = (
        shortness
        * rank_factor
        * preference_factor
        * scarcity
        * degree_balance
        * neighborhood_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)

    maximum = float(np.max(scores, initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float)
