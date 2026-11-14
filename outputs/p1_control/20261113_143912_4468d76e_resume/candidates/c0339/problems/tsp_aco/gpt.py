import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
        copy=True,
    )
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(valid, d, np.inf)

    # Use the second-nearest distance when possible as a robust local scale.
    kth = min(1, n - 2)
    local_scale = np.partition(safe, kth=kth, axis=1)[:, kth]

    row_counts = np.sum(valid, axis=1)
    row_sums = np.sum(np.where(valid, d, 0.0), axis=1)
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
        10.0 * fallback_distance,
    )
    safe_distance = np.maximum(safe_distance, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    row_ratio = safe_distance / local_scale[:, None]
    col_ratio = safe_distance / local_scale[None, :]
    mutual_ratio = 0.5 * (row_ratio + col_ratio)
    endpoint_ratio = np.sqrt(row_ratio * col_ratio)

    local_preference = np.exp(
        -1.20 * np.minimum(mutual_ratio, 16.0)
    )
    endpoint_preference = np.exp(
        -0.90 * np.minimum(endpoint_ratio, 16.0)
    )

    # Continuous approximation of mutual neighborhood preference.
    reciprocal_support = np.exp(
        -1.10 * np.minimum(
            np.abs(np.log1p(row_ratio) - np.log1p(col_ratio)),
            12.0,
        )
    )
    neighborhood_preference = (
        0.62
        + 0.38
        * reciprocal_support
        * np.exp(-0.28 * np.minimum(mutual_ratio, 12.0))
    )

    global_preference = 0.70 + 0.30 * np.exp(
        -0.24 * np.minimum(safe_distance / global_scale, 40.0)
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    balance = 1.0 / (
        1.0 + 0.80 * scale_difference / (scale_sum + eps)
    )
    neighborhood_balance = 0.80 + 0.20 * balance

    # Penalize edges long relative to both endpoints, while preserving bridges.
    endpoint_longness = np.maximum(row_ratio, col_ratio)
    excess_longness = np.maximum(endpoint_longness - 1.25, 0.0)
    bridge_factor = np.exp(-0.12 * np.minimum(excess_longness, 20.0))

    score = (
        local_preference
        * endpoint_preference
        * neighborhood_preference
        * global_preference
        * neighborhood_balance
        * bridge_factor
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
