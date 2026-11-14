import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = np.isfinite(d) & (d > 0)
    valid_edges = positive & positive.T

    if not np.any(valid_edges):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]

    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(positive, d, np.inf)
    sorted_edges = np.sort(safe, axis=1)

    second_index = min(1, n - 2)
    local_scale = sorted_edges[:, second_index]

    finite_local = np.isfinite(local_scale)
    if np.any(~finite_local):
        row_counts = np.sum(positive, axis=1)
        row_medians = np.full(n, global_scale, dtype=float)

        for i in np.flatnonzero(~finite_local):
            values = d[i, positive[i]]
            if values.size:
                row_medians[i] = np.median(values)

        local_scale = np.where(
            finite_local,
            local_scale,
            np.where(row_counts > 0, row_medians, fallback_distance),
        )

    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    relative_length = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1)

    mutual_rank = rank + rank.T

    exponential_proximity = np.exp(-1.00 * relative_length)
    reciprocal_proximity = 1.0 / (1.0 + relative_length)
    endpoint_preference = (
        0.60 * exponential_proximity
        + 0.40 * reciprocal_proximity
    )

    rank_preference = np.exp(-2.85 * mutual_rank)

    isolation = np.clip(local_scale / global_scale, 0.50, 2.50)
    isolation_pair = np.sqrt(isolation[:, None] * isolation[None, :])
    adaptive_distance = (
        safe_distance
        / global_scale
        / (0.78 + 0.22 * isolation_pair)
    )
    raw_global_preference = np.exp(-0.24 * adaptive_distance)
    global_preference = 0.68 + 0.32 * raw_global_preference

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    raw_balance = 1.0 / (
        1.0 + 0.75 * scale_difference / (scale_sum + eps)
    )
    neighborhood_balance = 0.80 + 0.20 * raw_balance

    neighborhood_kernel = np.where(
        positive,
        np.exp(-0.85 * d / np.maximum(local_scale[:, None], eps)),
        0.0,
    )
    kernel_row_sums = np.sum(neighborhood_kernel, axis=1, keepdims=True)
    normalized_kernel = neighborhood_kernel / np.maximum(
        kernel_row_sums,
        eps,
    )

    shared_support = normalized_kernel @ normalized_kernel.T
    support_values = shared_support[valid_edges]

    if support_values.size:
        support_min = float(np.min(support_values))
        support_max = float(np.max(support_values))
        support_range = max(support_max - support_min, eps)
        normalized_support = (
            shared_support - support_min
        ) / support_range
    else:
        normalized_support = np.zeros((n, n), dtype=float)

    structural_support = 0.90 + 0.10 * np.clip(
        normalized_support,
        0.0,
        1.0,
    )

    score = (
        endpoint_preference
        * rank_preference
        * global_preference
        * neighborhood_balance
        * structural_support
    )

    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
