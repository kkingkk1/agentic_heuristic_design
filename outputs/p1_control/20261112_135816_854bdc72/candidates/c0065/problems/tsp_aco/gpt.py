import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    positive = finite & (d > 0)

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    fallback = np.max(d[positive])
    d_safe = np.where(finite, d, fallback * 10.0)
    d_safe = np.maximum(d_safe, np.finfo(float).eps)

    row_sorted = np.sort(d_safe, axis=1)
    local_scale = row_sorted[:, min(2, n - 2)]
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_length = d_safe / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    absolute_scale = np.median(d_safe[positive])
    absolute_scale = max(absolute_scale, np.finfo(float).eps)

    row_order = np.argsort(d_safe, axis=1)
    rank = np.empty_like(row_order, dtype=float)
    rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]
    rank /= max(n - 1, 1)

    endpoint_preference = np.exp(-1.20 * relative_length)
    rank_preference = np.exp(-3.50 * (rank + rank.T))
    scale_preference = np.exp(-0.75 * d_safe / absolute_scale)

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    neighborhood_balance = 1.0 / (
        1.0 + 1.35 * scale_difference
        / (scale_sum + np.finfo(float).eps)
    )

    score = (
        endpoint_preference
        * rank_preference
        * scale_preference
        * neighborhood_balance
    )

    valid_edges = positive & positive.T
    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
