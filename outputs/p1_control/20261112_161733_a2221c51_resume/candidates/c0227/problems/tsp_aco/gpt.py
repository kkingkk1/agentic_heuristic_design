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

    valid = np.isfinite(d) & (d > 0)
    valid_symmetric = valid & valid.T

    if not np.any(valid_symmetric):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    fallback = float(np.max(positive_values))
    eps = np.finfo(float).eps

    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, eps)

    order = np.argsort(safe_d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(safe_d, order, axis=1)

    kth = min(2, n - 2)
    local_scale = sorted_d[:, kth]

    finite_counts = valid.sum(axis=1)
    row_min = np.min(np.where(valid, d, np.inf), axis=1)
    row_min = np.where(np.isfinite(row_min), row_min, fallback)
    local_scale = np.where(finite_counts > kth, local_scale, row_min)
    local_scale = np.maximum(local_scale, eps)

    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], order] = (
        np.arange(n, dtype=float)[None, :] / max(n - 1, 1)
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = safe_d / np.maximum(pair_scale, eps)

    rank_sum = row_rank + row_rank.T
    rank_gap = np.abs(row_rank - row_rank.T)

    endpoint_compatibility = np.exp(-1.28 * relative_distance)
    mutual_rank_preference = np.exp(
        -3.85 * rank_sum - 0.75 * rank_gap
    )

    global_scale = float(np.median(positive_values))
    global_scale = max(global_scale, eps)
    absolute_preference = np.exp(-0.48 * safe_d / global_scale)

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    balance_ratio = scale_difference / np.maximum(scale_sum, eps)
    neighborhood_balance = np.exp(-2.05 * balance_ratio)

    score = (
        endpoint_compatibility
        * mutual_rank_preference
        * absolute_preference
        * neighborhood_balance
    )

    score[~valid_symmetric] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
