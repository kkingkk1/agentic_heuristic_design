import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a conservative symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    valid_raw = np.isfinite(distances) & (distances >= 0.0) & ~diagonal
    raw = np.where(valid_raw, distances, np.inf)

    forward_valid = np.isfinite(raw)
    reverse_valid = np.isfinite(raw.T)
    valid = forward_valid | reverse_valid

    distance = np.full((n, n), np.inf, dtype=float)

    both_valid = forward_valid & reverse_valid
    distance[both_valid] = 0.5 * (raw[both_valid] + raw.T[both_valid])

    only_forward = forward_valid & ~reverse_valid
    distance[only_forward] = raw[only_forward]

    only_reverse = reverse_valid & ~forward_valid
    distance[only_reverse] = raw.T[only_reverse]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    local_k = min(5, n - 1)
    positive_local = np.where(
        sorted_valid & (sorted_distance > 0.0),
        sorted_distance,
        np.inf,
    )

    local_values = positive_local[:, :local_k]
    local_scale = np.median(
        np.where(np.isfinite(local_values), local_values, global_scale),
        axis=1,
    )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.05 * global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    relative_distance = np.divide(
        distance,
        pair_scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    global_ratio = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    global_ratio = np.clip(global_ratio, 0.0, 50.0)

    local_logits_i = -0.72 * relative_distance - 0.10 * np.divide(
        distance,
        local_scale[:, None],
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    local_logits_j = -0.72 * relative_distance - 0.10 * np.divide(
        distance,
        local_scale[None, :],
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    local_logits_i = np.where(valid, np.clip(local_logits_i, -50.0, 0.0), -np.inf)
    local_logits_j = np.where(valid, np.clip(local_logits_j, -50.0, 0.0), -np.inf)

    has_i = np.any(valid, axis=1, keepdims=True)
    has_j = np.any(valid, axis=1, keepdims=True)

    max_i = np.max(np.where(valid, local_logits_i, -np.inf), axis=1, keepdims=True)
    max_j = np.max(np.where(valid, local_logits_j, -np.inf), axis=1, keepdims=True)

    max_i = np.where(has_i, max_i, 0.0)
    max_j = np.where(has_j, max_j, 0.0)

    preferences_i = np.exp(np.clip(local_logits_i - max_i, -40.0, 0.0))
    preferences_j = np.exp(np.clip(local_logits_j - max_j, -40.0, 0.0))

    preferences_i = np.where(valid, preferences_i, 0.0)
    preferences_j = np.where(valid, preferences_j, 0.0)

    preferences_i /= np.maximum(preferences_i.sum(axis=1, keepdims=True), eps)
    preferences_j /= np.maximum(preferences_j.sum(axis=1, keepdims=True), eps)

    mutual_preference = np.sqrt(preferences_i * preferences_j.T)

    ranks = np.empty_like(order)
    rank_positions = np.broadcast_to(
        np.arange(n, dtype=order.dtype),
        order.shape,
    )
    np.put_along_axis(ranks, order, rank_positions, axis=1)

    rank_weight = np.where(
        valid,
        1.0 / (1.0 + ranks.astype(float)),
        0.0,
    )
    rank_agreement = np.sqrt(rank_weight * rank_weight.T)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.2 * np.sqrt(n)))),
    )

    top_indices = order[:, :neighborhood_size]
    top_valid = np.take_along_axis(valid, top_indices, axis=1)

    top_mask = np.zeros((n, n), dtype=bool)
    top_mask[
        np.arange(n)[:, None],
        top_indices,
    ] = top_valid

    mutual_top = (top_mask & top_mask.T & valid).astype(float)

    local_shortness = np.exp(-0.68 * relative_distance)
    global_shortness = np.exp(-0.40 * global_ratio)

    base = (
        0.50 * local_shortness
        + 0.32 * mutual_preference
        + 0.18 * global_shortness
    )

    scores = base * (0.82 + 0.18 * rank_agreement)
    scores *= 1.0 + 0.08 * mutual_top
    scores = np.where(valid, scores, 0.0)

    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)

    positive_scores = scores[scores > 0.0]
    if positive_scores.size:
        normalized = positive_scores / max(float(np.max(positive_scores)), eps)
        entropy = -float(np.mean(normalized * np.log(np.maximum(normalized, eps))))
        entropy_scale = max(float(np.log(max(2, positive_scores.size))), 1.0)
        concentration = np.clip(1.0 - entropy / entropy_scale, 0.0, 1.0)
        smoothing_power = 1.0 - 0.22 * concentration
        scores = np.power(np.maximum(scores, 0.0), smoothing_power)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
