import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized heuristic matrix for stochastic TSP sampling."""
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
    values = distance[valid]
    positive_values = values[values > 0.0]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.median(values))
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    local_k = min(
        n - 1,
        max(3, int(np.ceil(1.2 * np.sqrt(n)))),
    )

    local_values = sorted_distance[:, :local_k]
    local_valid = sorted_valid[:, :local_k]
    local_values = np.where(local_valid, local_values, global_scale)

    local_scale = np.median(local_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale >= 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.05 * global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    relative_local = np.divide(
        distance,
        pair_scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_local = np.clip(relative_local, 0.0, 50.0)

    relative_global = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_global = np.clip(relative_global, 0.0, 50.0)

    local_component = np.divide(
        distance,
        local_scale[:, None],
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    local_component = np.clip(local_component, 0.0, 50.0)

    logits = (
        -0.72 * relative_local
        -0.10 * local_component
    )
    logits = np.where(valid, np.clip(logits, -50.0, 0.0), -np.inf)

    row_max = np.max(
        np.where(valid, logits, -np.inf),
        axis=1,
        keepdims=True,
    )
    has_neighbor = np.any(valid, axis=1, keepdims=True)
    row_max = np.where(has_neighbor, row_max, 0.0)

    stabilized = np.where(
        valid,
        np.clip(logits - row_max, -40.0, 0.0),
        -np.inf,
    )

    sharp_weights = np.where(valid, np.exp(stabilized), 0.0)
    sharp_sums = sharp_weights.sum(axis=1, keepdims=True)
    sharp_preferences = np.divide(
        sharp_weights,
        np.maximum(sharp_sums, eps),
        out=np.zeros_like(sharp_weights),
        where=sharp_sums > 0.0,
    )

    broad_weights = np.where(
        valid,
        np.exp(0.55 * stabilized),
        0.0,
    )
    broad_sums = broad_weights.sum(axis=1, keepdims=True)
    broad_preferences = np.divide(
        broad_weights,
        np.maximum(broad_sums, eps),
        out=np.zeros_like(broad_weights),
        where=broad_sums > 0.0,
    )

    entropy = -np.sum(
        np.where(
            sharp_preferences > 0.0,
            sharp_preferences
            * np.log(np.maximum(sharp_preferences, eps)),
            0.0,
        ),
        axis=1,
    )

    degree = valid.sum(axis=1).astype(float)
    entropy_limit = np.log(np.maximum(degree, 2.0))
    normalized_entropy = np.divide(
        entropy,
        np.maximum(entropy_limit, eps),
    )
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

    exploration = 0.10 + 0.28 * (1.0 - normalized_entropy)
    calibrated = (
        (1.0 - exploration[:, None]) * sharp_preferences
        + exploration[:, None] * broad_preferences
    )

    calibrated = np.where(valid, calibrated, 0.0)
    calibrated /= np.maximum(
        calibrated.sum(axis=1, keepdims=True),
        eps,
    )

    mutual_preference = np.sqrt(
        np.maximum(calibrated, 0.0)
        * np.maximum(calibrated.T, 0.0)
    )

    ranks = np.empty_like(order, dtype=float)
    rank_positions = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    np.put_along_axis(
        ranks,
        order,
        rank_positions,
        axis=1,
    )

    rank_weight = np.where(
        valid,
        1.0 / (1.0 + ranks),
        0.0,
    )
    rank_agreement = np.sqrt(
        rank_weight * rank_weight.T
    )

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.5 * np.sqrt(n)))),
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

    transition = calibrated.copy()
    np.fill_diagonal(transition, 0.0)

    continuation = transition @ transition
    continuation_scale = np.max(
        continuation,
        axis=1,
        keepdims=True,
    )
    continuation = np.divide(
        continuation,
        np.maximum(
            np.sqrt(
                continuation_scale
                * np.maximum(continuation_scale.T, eps)
            ),
            eps,
        ),
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    mutual_continuation = np.sqrt(
        np.maximum(continuation, 0.0)
        * np.maximum(continuation.T, 0.0)
    )

    valid_counts = valid.sum(axis=1).astype(float)
    scarcity = 1.0 - np.divide(
        np.minimum(valid_counts, float(neighborhood_size)),
        max(float(neighborhood_size), 1.0),
    )
    scarcity = np.clip(scarcity, 0.0, 1.0)

    uniform = np.divide(
        valid.astype(float),
        np.maximum(valid_counts[:, None], 1.0),
    )
    diversity_floor = 0.015 + 0.045 * np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    local_shortness = np.exp(-0.68 * relative_local)
    global_shortness = np.exp(-0.35 * relative_global)

    scores = (
        0.54 * local_shortness
        + 0.27 * mutual_preference
        + 0.10 * global_shortness
        + 0.05 * mutual_continuation
    )

    scores *= (
        0.96
        + 0.04 * rank_agreement
        + 0.02 * mutual_top
    )

    scores += diversity_floor * (
        0.75 * uniform
        + 0.25 * global_shortness
    )

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
