import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
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

    valid_forward = np.isfinite(raw)
    valid_reverse = np.isfinite(reverse)
    both_valid = valid_forward & valid_reverse

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )

    only_forward = valid_forward & ~valid_reverse
    only_reverse = valid_reverse & ~valid_forward
    distance[only_forward] = raw[only_forward]
    distance[only_reverse] = reverse[only_reverse]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return result

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

    candidate_count = valid.sum(axis=1).astype(float)
    safe_distance = np.where(valid, distance, np.inf)

    order = np.argsort(safe_distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(safe_distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    positive_sorted = np.where(
        sorted_valid & (sorted_distance > 0.0),
        sorted_distance,
        np.inf,
    )

    local_k = min(5, n - 1)
    local_values = positive_sorted[:, :local_k]

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

    local_logit_i = (
        -0.72 * relative_distance
        -0.18 * endpoint_ratio_i
    )
    local_logit_j = (
        -0.72 * relative_distance
        -0.18 * endpoint_ratio_j
    )

    local_logit_i = np.where(valid, local_logit_i, -np.inf)
    local_logit_j = np.where(valid, local_logit_j, -np.inf)

    max_i = np.max(local_logit_i, axis=1, keepdims=True)
    max_j = np.max(local_logit_j, axis=1, keepdims=True)

    weights_i = np.exp(np.clip(local_logit_i - max_i, -40.0, 0.0))
    weights_j = np.exp(np.clip(local_logit_j - max_j, -40.0, 0.0))

    weights_i = np.where(valid, weights_i, 0.0)
    weights_j = np.where(valid, weights_j, 0.0)

    sum_i = np.maximum(weights_i.sum(axis=1, keepdims=True), eps)
    sum_j = np.maximum(weights_j.sum(axis=1, keepdims=True), eps)

    preferences_i = weights_i / sum_i
    preferences_j = weights_j / sum_j

    mutual_preference = np.sqrt(
        preferences_i * preferences_j.T
    )

    rank_weight = np.zeros((n, n), dtype=float)
    rank_weight[valid] = 1.0 / (
        1.0
        + np.take_along_axis(
            np.argsort(np.argsort(safe_distance, axis=1, kind="stable"),
                       axis=1, kind="stable"),
            np.arange(n)[None, :].repeat(n, axis=0),
            axis=1,
        )[valid]
    )

    rank_agreement = np.sqrt(
        rank_weight * rank_weight.T
    )

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.2 * np.sqrt(n)))),
    )

    top_mask = np.zeros((n, n), dtype=bool)
    top_indices = order[:, :neighborhood_size]
    top_valid = np.take_along_axis(
        valid,
        top_indices,
        axis=1,
    )
    top_mask[
        np.arange(n)[:, None],
        top_indices,
    ] = top_valid

    mutual_top = (top_mask & top_mask.T & valid).astype(float)
    mutual_nearest = (
        (top_mask[:, :].sum(axis=1) > 0)
        & (top_mask.T.sum(axis=1) > 0)
    )

    nearest = np.zeros((n, n), dtype=float)
    nearest_indices = order[:, 0]
    nearest_valid = valid[np.arange(n), nearest_indices]
    nearest[np.arange(n), nearest_indices] = nearest_valid.astype(float)

    mutual_nearest = nearest * nearest.T

    shortness = np.exp(-0.82 * relative_distance)
    endpoint_affinity = np.exp(
        -0.24 * (endpoint_ratio_i + endpoint_ratio_j)
    )

    scarcity = np.divide(
        np.minimum(local_scale[:, None], local_scale[None, :]),
        distance + 0.5 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    scarcity = 1.0 + 0.08 * np.clip(scarcity, 0.0, 2.0)

    density = np.sqrt(
        np.clip(
            candidate_count[:, None]
            * candidate_count[None, :]
            / max(float(n - 1) ** 2, 1.0),
            0.5,
            1.0,
        )
    )

    base = (
        0.58 * shortness
        + 0.42 * mutual_preference
    )

    scores = (
        base
        * endpoint_affinity
        * (0.72 + 0.28 * rank_agreement)
        * (1.0 + 0.16 * mutual_top)
        * (1.0 + 0.08 * mutual_nearest)
        * scarcity
        * (0.92 + 0.08 * density)
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
