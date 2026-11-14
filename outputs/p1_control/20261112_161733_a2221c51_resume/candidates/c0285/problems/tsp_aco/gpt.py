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
    d = (d + d.T) * 0.5
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(valid, d, np.inf)

    kth = min(1, n - 2)
    local_scale = np.partition(safe, kth=kth, axis=1)[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    invalid_distance = fallback_distance * 10.0
    safe_distance = np.where(valid, d, invalid_distance)
    safe_distance = np.maximum(safe_distance, eps)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scale_product = np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        )
        relative_length = safe_distance / scale_product

        endpoint_preference = np.exp(-1.10 * relative_length)

        order = np.argsort(
            safe_distance,
            axis=1,
            kind="stable",
        )
        rank = np.empty((n, n), dtype=float)
        row_index = np.arange(n)[:, None]
        rank[row_index, order] = np.arange(n, dtype=float)[None, :]
        rank /= max(n - 1, 1)

        mutual_rank = rank + rank.T
        rank_preference = np.exp(-3.20 * mutual_rank)

        global_preference = 0.62 + 0.38 * np.exp(
            -0.26 * safe_distance / global_scale
        )

        local_i = local_scale[:, None]
        local_j = local_scale[None, :]
        scale_difference = np.abs(local_i - local_j)
        scale_sum = local_i + local_j

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

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
