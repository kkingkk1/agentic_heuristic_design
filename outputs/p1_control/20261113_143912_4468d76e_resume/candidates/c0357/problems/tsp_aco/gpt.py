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
    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(positive, d, np.inf)

    kth = 0 if n <= 2 else 1
    local_scale = np.partition(safe, kth, axis=1)[:, kth]

    row_counts = np.sum(positive, axis=1)
    row_sums = np.sum(np.where(positive, d, 0.0), axis=1)
    row_means = np.divide(
        row_sums,
        np.maximum(row_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=row_counts > 0,
    )

    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        row_means,
    )
    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    row_ratio = safe_distance / local_scale[:, None]
    col_ratio = safe_distance / local_scale[None, :]
    mutual_ratio = 0.5 * (row_ratio + col_ratio)
    endpoint_ratio = np.sqrt(row_ratio * col_ratio)

    local_preference = np.exp(-1.25 * np.minimum(mutual_ratio, 12.0))
    endpoint_preference = np.exp(-0.75 * np.minimum(endpoint_ratio, 12.0))

    global_preference = 0.70 + 0.30 * np.exp(
        -0.22 * np.minimum(safe_distance / global_scale, 40.0)
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    balance = 1.0 / (
        1.0 + 0.65 * scale_difference / (scale_sum + eps)
    )
    neighborhood_balance = 0.78 + 0.22 * balance

    score = (
        local_preference
        * endpoint_preference
        * global_preference
        * neighborhood_balance
    )

    valid_edges = positive & positive.T
    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
