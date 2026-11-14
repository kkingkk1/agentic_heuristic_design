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
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    values = d[valid]

    global_scale = max(float(np.median(values)), eps)
    fallback = max(float(np.max(values)), global_scale)

    safe = np.where(valid, d, np.inf)

    kth = min(2, n - 1)
    local_scale = np.partition(safe, kth=kth, axis=1)[:, kth]

    invalid_local = ~np.isfinite(local_scale)
    if np.any(invalid_local):
        for i in np.flatnonzero(invalid_local):
            row_values = d[i, valid[i]]
            local_scale[i] = (
                max(float(np.median(row_values)), eps)
                if row_values.size
                else fallback
            )

    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    blended_scale = np.sqrt(
        0.72 * local_scale[:, None] * local_scale[None, :]
        + 0.28 * global_scale * global_scale
    )
    relative_length = safe_distance / np.maximum(blended_scale, eps)

    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1)

    mutual_rank = rank + rank.T
    local_signal = np.exp(
        -1.08 * relative_length - 2.25 * mutual_rank
    )

    global_signal = np.exp(
        -0.62 * safe_distance / global_scale
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    neighborhood_balance = 1.0 / (
        1.0 + 1.15 * scale_difference / (scale_sum + eps)
    )

    score = (
        (0.84 * local_signal + 0.16 * global_signal)
        * neighborhood_balance
    )

    score[~(valid & valid.T)] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
