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

    eps = np.finfo(float).eps
    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
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

    local_logits = (
        -0.72 * relative_distance
        -0.10
        * np.divide(
            distance,
            local_scale[:, None],
            out=np.full((n, n), 50.0, dtype=float),
            where=valid,
        )
    )
    local_logits = np.where(
        valid,
        np.clip(local_logits, -50.0, 0.0),
        -np.inf,
    )

    has_neighbor = np.any(valid, axis=1, keepdims=True)
    row_max = np.max(
        np.where(valid, local_logits, -np.inf),
        axis=1,
        keepdims=True,
    )
    row_max = np.where(has_neighbor, row_max, 0.0)

    preferences = np.exp(
        np.clip(local_logits - row_max, -40.0, 0.0)
    )
    preferences = np.where(valid, preferences, 0.0)
    preferences /= np.maximum(
        preferences.sum(axis=1, keepdims=True),
        eps,
    )

    # Estimate concentration separately for each node.
    degree = valid.sum(axis=1).astype(float)
    entropy = -np.sum(
        preferences * np.log(np.maximum(preferences, eps)),
        axis=1,
    )
    entropy_limit = np.log(np.maximum(degree, 2.0))
    normalized_entropy = np.divide(
        entropy,
        entropy_limit,
        out=np.zeros_like(entropy),
        where=entropy_limit > 0.0,
    )
    concentration = np.clip(1.0 - normalized_entropy, 0.0, 1.0)

    # Smooth only concentrated nodes, preserving the local ordering.
    node_power = 1.0 - 0.18 * concentration
    calibrated = np.power(
        np.maximum(preferences, 0.0),
        node_power[:, None],
    )
    calibrated = np.where(valid, calibrated, 0.0)
    calibrated /= np.maximum(
        calibrated.sum(axis=1, keepdims=True),
        eps,
    )

    mutual_preference = np.sqrt(
        calibrated * calibrated.T
    )

    ranks = np.empty_like(order)
    rank_positions = np.broadcast_to(
        np.arange(n, dtype=order.dtype),
        order.shape,
    )
    np.put_along_axis(
        ranks,
        order,
        rank_positions,
        axis=1,
    )

    rank_weight = np.where(
        valid,
        1.0 / (1.0 + ranks.astype(float)),
        0.0,
    )
    rank_agreement = np.sqrt(
        rank_weight * rank_weight.T
    )

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

    mutual_top = (
        top_mask
        & top_mask.T
        & valid
    ).astype(float)

    local_shortness = np.exp(-0.68 * relative_distance)
    global_shortness = np.exp(-0.40 * global_ratio)

    base = (
        0.54 * local_shortness
        + 0.30 * mutual_preference
        + 0.16 * global_shortness
    )

    scores = base * (0.94 + 0.06 * rank_agreement)
    scores *= 1.0 + 0.06 * mutual_top
    scores = np.where(valid, scores, 0.0)

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
