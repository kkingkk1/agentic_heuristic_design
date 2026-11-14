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
    valid &= valid.T

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

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        global_short = 1.0 / (
            1.0 + safe_distance / global_scale
        )

        relative_length = safe_distance / np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        )
        local_short = 1.0 / (1.0 + relative_length)

        order = np.argsort(safe_distance, axis=1, kind="stable")
        rank = np.empty((n, n), dtype=float)
        rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
        rank /= max(n - 1, 1)

        mutual_rank = rank + rank.T
        mutual_short = 1.0 / (1.0 + 2.0 * mutual_rank)

        scale_difference = np.abs(
            local_scale[:, None] - local_scale[None, :]
        )
        scale_sum = local_scale[:, None] + local_scale[None, :]
        density_balance = 1.0 / (
            1.0 + scale_difference / (scale_sum + eps)
        )

        local_with_floor = (
            (0.62 + 0.18 * density_balance) * local_short
            + (0.38 - 0.18 * density_balance) * global_short
        )

        score = (
            0.42 * global_short
            + 0.30 * mutual_short
            + 0.28 * local_with_floor
        )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
