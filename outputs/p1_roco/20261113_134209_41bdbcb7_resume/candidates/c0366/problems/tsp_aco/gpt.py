import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized heuristic matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    valid_input = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    raw = np.where(valid_input, distances, np.inf)
    reverse = raw.T

    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)
    both_valid = forward_valid & backward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (raw[both_valid] + reverse[both_valid])

    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = reverse[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return result

    values = distance[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    valid_counts = valid.sum(axis=1).astype(float)
    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.5 * np.sqrt(n)))),
    )

    neighborhood = sorted_distance[:, :neighborhood_size]
    neighborhood_valid = sorted_valid[:, :neighborhood_size]

    neighborhood_values = np.where(
        neighborhood_valid,
        neighborhood,
        global_scale,
    )

    local_scale = np.median(neighborhood_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale >= 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.01 * global_scale)

    relative_local = np.divide(
        distance,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    relative_global = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    relative_local = np.clip(relative_local, 0.0, 50.0)
    relative_global = np.clip(relative_global, 0.0, 50.0)

    local_shortness = np.exp(-0.72 * relative_local)
    global_shortness = np.exp(-0.30 * relative_global)

    ranks = np.zeros((n, n), dtype=float)
    rank_values = np.where(
        sorted_valid,
        np.arange(1, n + 1, dtype=float)[None, :],
        0.0,
    )
    np.put_along_axis(
        ranks,
        order,
        np.divide(
            1.0,
            np.maximum(rank_values, 1.0),
            out=np.zeros_like(rank_values),
            where=sorted_valid,
        ),
        axis=1,
    )

    mutual_rank = np.sqrt(
        np.maximum(ranks, 0.0) * np.maximum(ranks.T, 0.0)
    )

    positive_all = values[values > 0.0]
    if positive_all.size:
        q25, q50, q75 = np.percentile(
            positive_all,
            [25.0, 50.0, 75.0],
        )
    else:
        q25, q50, q75 = 0.0, global_scale, global_scale

    spread = max(float(q75 - q25), 0.1 * global_scale, eps)

    global_rank_signal = 1.0 / (
        1.0 + np.maximum(distance - q50, 0.0) / spread
    )
    global_rank_signal = np.where(valid, global_rank_signal, 0.0)

    base_logits = -0.78 * relative_local - 0.18 * relative_global
    base_logits = np.where(valid, base_logits, -np.inf)

    row_max = np.max(
        np.where(valid, base_logits, -np.inf),
        axis=1,
        keepdims=True,
    )
    stabilized = np.where(
        valid,
        base_logits - row_max,
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
            sharp_preferences * np.log(
                np.maximum(sharp_preferences, eps)
            ),
            0.0,
        ),
        axis=1,
    )

    max_entropy = np.log(np.maximum(valid_counts, 2.0))
    normalized_entropy = np.divide(
        entropy,
        np.maximum(max_entropy, eps),
    )
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

    exploration = 0.12 + 0.38 * (1.0 - normalized_entropy)
    calibrated_preferences = (
        (1.0 - exploration[:, None]) * sharp_preferences
        + exploration[:, None] * broad_preferences
    )

    mutual_preference = np.sqrt(
        np.maximum(calibrated_preferences, 0.0)
        * np.maximum(calibrated_preferences.T, 0.0)
    )

    transition = calibrated_preferences.copy()
    np.fill_diagonal(transition, 0.0)

    continuation = transition @ transition
    continuation_scale = np.max(continuation, axis=1, keepdims=True)
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

    scarcity = 1.0 - np.divide(
        np.minimum(valid_counts, float(neighborhood_size)),
        max(float(neighborhood_size), 1.0),
    )
    scarcity = np.clip(scarcity, 0.0, 1.0)

    diversity_floor = 0.025 + 0.075 * np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    uniform = np.divide(
        valid.astype(float),
        np.maximum(valid_counts[:, None], 1.0),
    )

    scores = (
        0.48 * local_shortness
        + 0.16 * mutual_preference
        + 0.13 * mutual_continuation
        + 0.12 * mutual_rank
        + 0.07 * global_shortness
        + 0.04 * global_rank_signal
    )

    scores += diversity_floor * (
        0.70 * uniform
        + 0.30 * global_rank_signal
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
