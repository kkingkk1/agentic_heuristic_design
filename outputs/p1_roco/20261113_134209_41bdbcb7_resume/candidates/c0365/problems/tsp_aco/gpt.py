import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a robust, symmetric, normalized TSP edge-priority matrix."""
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

    forward_valid = np.isfinite(raw)
    reverse_valid = np.isfinite(raw.T)
    valid = forward_valid | reverse_valid

    distance = np.full((n, n), np.inf, dtype=float)

    both_valid = forward_valid & reverse_valid
    distance[both_valid] = 0.5 * (raw[both_valid] + raw.T[both_valid])

    only_forward = forward_valid & ~reverse_valid
    only_reverse = reverse_valid & ~forward_valid
    distance[only_forward] = raw[only_forward]
    distance[only_reverse] = raw.T[only_reverse]

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
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    q25, q75 = np.percentile(finite_values, [25.0, 75.0])
    spread = max(float(q75 - q25), 0.1 * global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    local_k = min(n - 1, max(3, int(np.ceil(1.3 * np.sqrt(n)))))
    local_positive = np.where(
        sorted_valid[:, :local_k] & (sorted_distance[:, :local_k] > 0.0),
        sorted_distance[:, :local_k],
        np.inf,
    )

    local_scale = np.median(
        np.where(np.isfinite(local_positive), local_positive, global_scale),
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

    relative_local = np.divide(
        distance,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    relative_local = np.clip(relative_local, 0.0, 50.0)

    relative_global = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    relative_global = np.clip(relative_global, 0.0, 50.0)

    local_shortness = np.exp(-0.64 * relative_local)
    global_shortness = np.exp(-0.34 * relative_global)

    logits = (
        -0.52 * relative_local
        -0.10 * np.minimum(
            np.divide(
                distance,
                local_scale[:, None],
                out=np.full((n, n), np.inf, dtype=float),
                where=valid,
            ),
            50.0,
        )
    )
    logits = np.where(valid, logits, -np.inf)

    row_max = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(logits - row_max, -40.0, 0.0))
    weights = np.where(valid, weights, 0.0)

    row_sum = np.maximum(weights.sum(axis=1, keepdims=True), eps)
    preferences = weights / row_sum
    mutual_preference = np.sqrt(
        np.maximum(preferences, 0.0)
        * np.maximum(preferences.T, 0.0)
    )

    ranks = np.empty_like(order, dtype=float)
    rank_positions = np.broadcast_to(
        np.arange(n, dtype=float),
        order.shape,
    )
    np.put_along_axis(ranks, order, rank_positions, axis=1)

    rank_weight = np.where(
        valid,
        1.0 / (1.0 + ranks),
        0.0,
    )
    rank_agreement = np.sqrt(
        np.maximum(rank_weight, 0.0)
        * np.maximum(rank_weight.T, 0.0)
    )

    first = np.where(
        sorted_valid[:, 0],
        sorted_distance[:, 0],
        global_scale,
    )
    second = np.where(
        sorted_valid[:, 1],
        sorted_distance[:, 1],
        first,
    )
    third = np.where(
        sorted_valid[:, 2],
        sorted_distance[:, 2],
        second,
    ) if n > 2 else second

    first_gap = np.clip(
        (second - first) / np.maximum(local_scale, eps),
        0.0,
        4.0,
    )
    second_gap = np.clip(
        (third - second) / np.maximum(local_scale, eps),
        0.0,
        4.0,
    )

    regret = np.sqrt(
        (0.35 + first_gap[:, None])
        * (0.35 + first_gap[None, :])
    )
    regret *= np.exp(-0.20 * np.maximum(relative_local, 0.0))
    regret = np.clip(regret / 2.0, 0.0, 1.0)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.25 * np.sqrt(n)))),
    )
    top_indices = order[:, :neighborhood_size]
    top_valid = np.take_along_axis(valid, top_indices, axis=1)

    top_mask = np.zeros((n, n), dtype=bool)
    top_mask[
        np.arange(n)[:, None],
        top_indices,
    ] = top_valid

    mutual_top = (top_mask & top_mask.T & valid).astype(float)

    candidate_mask = valid & (
        (ranks <= 2.0)
        | (distance <= q75 + spread)
    )
    candidate_strength = np.where(
        candidate_mask,
        np.exp(-0.34 * relative_local),
        0.0,
    )

    continuation = candidate_strength @ candidate_strength
    row_strength = candidate_strength.sum(axis=1)
    denominator = np.sqrt(
        np.maximum(row_strength[:, None] * row_strength[None, :], eps)
    )
    continuation = np.divide(
        continuation,
        denominator,
        out=np.zeros_like(continuation),
        where=denominator > 0.0,
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    two_step = np.sqrt(
        np.maximum(continuation, 0.0)
        * np.maximum(continuation.T, 0.0)
    )

    bridge = 1.0 / (
        1.0 + np.exp(
            np.clip((distance - q25) / spread, -20.0, 20.0)
        )
    )

    valid_counts = valid.sum(axis=1).astype(float)
    branching = np.minimum(valid_counts, float(neighborhood_size))
    scarcity = 1.0 - branching / max(float(neighborhood_size), 1.0)
    scarcity_pair = np.sqrt(
        np.maximum(scarcity[:, None], 0.0)
        * np.maximum(scarcity[None, :], 0.0)
    )

    uniform = valid.astype(float) / np.maximum(
        valid_counts[:, None],
        1.0,
    )

    scores = (
        0.40 * local_shortness
        + 0.25 * mutual_preference
        + 0.13 * global_shortness
        + 0.08 * rank_agreement
        + 0.06 * regret
        + 0.05 * two_step
        + 0.03 * bridge
    )

    exploration = 0.025 + 0.045 * scarcity_pair
    scores += exploration * (
        0.60 * uniform
        + 0.25 * bridge
        + 0.15 * two_step
    )

    scores *= 1.0 + 0.06 * mutual_top
    scores += 0.025 * scarcity_pair * regret

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
