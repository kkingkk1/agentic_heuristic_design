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

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]

    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(positive, d, np.inf)

    second_index = min(1, n - 2)
    local_scale = np.partition(
        safe,
        kth=second_index,
        axis=1,
    )[:, second_index]

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

    endpoint_preference = np.exp(-1.10 * relative_length)
    rank_preference = np.exp(-3.20 * mutual_rank)

    raw_global_preference = np.exp(-0.26 * safe_distance / global_scale)
    global_preference = 0.68 + 0.32 * raw_global_preference

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    raw_balance = 1.0 / (
        1.0 + 0.85 * scale_difference / (scale_sum + eps)
    )
    neighborhood_balance = 0.78 + 0.22 * raw_balance

    score = (
        endpoint_preference
        * rank_preference
        * global_preference
        * neighborhood_balance
    )

    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
