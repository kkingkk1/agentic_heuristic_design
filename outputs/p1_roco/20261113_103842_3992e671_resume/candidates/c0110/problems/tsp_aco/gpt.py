import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    raw = distance_matrix.copy()
    finite_original = np.isfinite(raw)
    np.fill_diagonal(finite_original, False)

    positive = raw[finite_original & (raw > 0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    raw[~np.isfinite(raw)] = np.inf
    raw[raw < 0] = 0.0
    np.fill_diagonal(raw, np.inf)

    finite = np.isfinite(raw)
    finite_t = finite.T

    both_finite = finite & finite_t
    only_forward = finite & ~finite_t
    only_backward = finite_t & ~finite

    symmetric_distance = np.full((n, n), np.inf, dtype=float)
    symmetric_distance[both_finite] = (
        raw[both_finite] + raw.T[both_finite]
    ) * 0.5
    symmetric_distance[only_forward] = raw[only_forward]
    symmetric_distance[only_backward] = raw.T[only_backward]
    np.fill_diagonal(symmetric_distance, np.inf)

    valid = np.isfinite(symmetric_distance)
    valid_count = valid.sum(axis=1)

    if not np.any(valid):
        return scores

    sorted_distances = np.sort(
        np.where(valid, symmetric_distance, np.inf),
        axis=1,
    )

    lower_index = np.maximum((valid_count - 1) // 2, 0)
    upper_index = np.maximum(valid_count // 2, 0)
    row_median = 0.5 * (
        sorted_distances[np.arange(n), lower_index]
        + sorted_distances[np.arange(n), upper_index]
    )
    row_median[valid_count == 0] = global_scale

    row_min = sorted_distances[:, 0]
    row_min[valid_count == 0] = global_scale

    local_scale = np.maximum(
        0.5 * (row_median + row_min),
        np.finfo(float).eps,
    )

    normalized = symmetric_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    clipped = np.clip(normalized, 0.0, 80.0)
    logits = np.where(valid, -clipped, -np.inf)
    row_max = np.max(logits, axis=1)
    row_max = np.where(valid_count > 0, row_max, 0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        preference_weights = np.where(
            valid,
            np.exp(logits - row_max[:, None]),
            0.0,
        )

    preference_sum = preference_weights.sum(axis=1, keepdims=True)
    preferences = np.divide(
        preference_weights,
        preference_sum,
        out=np.zeros_like(preference_weights),
        where=preference_sum > 0,
    )

    bidirectional_preference = np.sqrt(
        preferences * preferences.T
    )

    order = np.argsort(
        np.where(valid, symmetric_distance, np.inf),
        axis=1,
    )
    inverse_order = np.argsort(order, axis=1)
    ranks = np.where(
        valid,
        1.0 / (1.0 + inverse_order.astype(float)),
        0.0,
    )

    rank_continuity = np.sqrt(ranks * ranks.T)

    nearest = (valid & (inverse_order == 0)).astype(float)
    mutual_nearest = nearest * nearest.T

    top_two = (
        valid & (inverse_order < np.minimum(valid_count[:, None], 2))
    ).astype(float)
    mutual_top_two = top_two * top_two.T

    second_index = np.minimum(np.maximum(valid_count - 1, 0), 1)
    second_nearest = sorted_distances[
        np.arange(n),
        second_index,
    ]
    second_nearest[valid_count == 0] = global_scale

    gap = np.maximum(second_nearest, global_scale)
    safe_distance = np.maximum(
        symmetric_distance,
        np.finfo(float).eps,
    )

    isolation = np.ones((n, n), dtype=float)
    isolation_increment = 0.25 * np.clip(
        gap[:, None] / safe_distance - 1.0,
        0.0,
        3.0,
    )
    isolation = np.where(
        valid,
        isolation + isolation_increment,
        1.0,
    )
    isolation_bonus = np.sqrt(isolation * isolation.T)

    scores = (
        bidirectional_preference
        * (0.60 + 0.40 * rank_continuity)
        * (1.0 + 0.80 * mutual_nearest + 0.20 * mutual_top_two)
        * isolation_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores
