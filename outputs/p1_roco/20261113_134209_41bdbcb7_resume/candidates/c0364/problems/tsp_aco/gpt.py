import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a conservative symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    raw = np.where(finite_nonnegative & ~diagonal, distances, np.inf)

    forward_valid = np.isfinite(raw)
    reverse_valid = np.isfinite(raw.T)
    valid = forward_valid | reverse_valid

    distance = np.full((n, n), np.inf, dtype=float)

    both_valid = forward_valid & reverse_valid
    distance[both_valid] = 0.5 * (
        raw[both_valid] + raw.T[both_valid]
    )

    only_forward = forward_valid & ~reverse_valid
    only_reverse = reverse_valid & ~forward_valid
    distance[only_forward] = raw[only_forward]
    distance[only_reverse] = raw.T[only_reverse]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return result

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    safe_distance = np.where(valid, distance, np.inf)
    order = np.argsort(safe_distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(safe_distance, order, axis=1)
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

    endpoint_ratio_i = np.divide(
        distance,
        local_scale[:, None],
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio_j = np.divide(
        distance,
        local_scale[None, :],
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    endpoint_ratio_i = np.clip(endpoint_ratio_i, 0.0, 50.0)
    endpoint_ratio_j = np.clip(endpoint_ratio_j, 0.0, 50.0)

    logits_i = (
        -0.58 * relative_distance
        -0.12 * endpoint_ratio_i
    )
    logits_j = (
        -0.58 * relative_distance
        -0.12 * endpoint_ratio_j
    )

    logits_i = np.where(valid, logits_i, -np.inf)
    logits_j = np.where(valid, logits_j, -np.inf)

    max_i = np.max(logits_i, axis=1, keepdims=True)
    max_j = np.max(logits_j, axis=1, keepdims=True)

    weights_i = np.exp(np.clip(logits_i - max_i, -40.0, 0.0))
    weights_j = np.exp(np.clip(logits_j - max_j, -40.0, 0.0))

    weights_i = np.where(valid, weights_i, 0.0)
    weights_j = np.where(valid, weights_j, 0.0)

    sum_i = np.maximum(weights_i.sum(axis=1, keepdims=True), eps)
    sum_j = np.maximum(weights_j.sum(axis=1, keepdims=True), eps)

    preferences_i = weights_i / sum_i
    preferences_j = weights_j / sum_j

    mutual_preference = np.sqrt(
        preferences_i * preferences_j.T
    )

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
    top_valid = np.take_along_axis(
        valid,
        top_indices,
        axis=1,
    )

    top_mask = np.zeros((n, n), dtype=bool)
    top_mask[
        np.arange(n)[:, None],
        top_indices,
    ] = top_valid

    mutual_top = (top_mask & top_mask.T & valid).astype(float)

    local_shortness = np.exp(-0.68 * relative_distance)
    global_shortness = np.exp(
        -0.42 * np.clip(distance / global_scale, 0.0, 50.0)
    )

    base = (
        0.48 * local_shortness
        + 0.32 * mutual_preference
        + 0.20 * global_shortness
    )

    scores = (
        base
        * (0.78 + 0.22 * rank_agreement)
        * (1.0 + 0.10 * mutual_top)
    )

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
