import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = (d + d.T) * 0.5
    finite_positive = np.isfinite(d) & (d > 0)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    fallback = float(np.max(positive_values))
    eps = np.finfo(float).eps
    replacement = max(fallback * 10.0, eps)

    valid = finite_positive & finite_positive.T

    work = np.full((n, n), replacement, dtype=float)
    np.copyto(work, d, where=finite_positive)
    np.fill_diagonal(work, replacement)

    kth = 0 if n <= 2 else 1
    local_scale = np.partition(work, kth=kth, axis=1)[:, kth]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.arange(n, dtype=order.dtype)[None, :],
        axis=1,
    )

    rank_sum = ranks + ranks.T
    log_score = -1.8 * np.log1p(rank_sum.astype(float))

    global_scale = max(float(np.median(positive_values)), eps)

    work /= global_scale
    log_score -= 0.35 * work

    work /= endpoint_scale
    log_score -= 0.9 * work

    local_sum = local_scale[:, None] + local_scale[None, :] + eps
    scale_difference = (
        np.abs(local_scale[:, None] - local_scale[None, :]) / local_sum
    )
    log_score -= 1.2 * scale_difference

    log_score -= np.max(log_score[valid])

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = (score + score.T) * 0.5

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
