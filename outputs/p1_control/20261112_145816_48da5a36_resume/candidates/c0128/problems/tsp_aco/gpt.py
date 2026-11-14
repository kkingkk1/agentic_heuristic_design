import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    finite_positive = np.isfinite(d) & (d > 0)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    fallback = float(np.max(positive_values))
    replacement = max(fallback * 10.0, np.finfo(float).eps)

    valid = finite_positive & finite_positive.T
    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    sorted_distances = np.sort(work, axis=1)
    kth = min(1, n - 2)
    local_scale = np.maximum(
        sorted_distances[:, kth],
        np.finfo(float).eps,
    )

    normalized_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    global_scale = max(float(np.median(positive_values)), np.finfo(float).eps)
    absolute_length = work / global_scale

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + np.finfo(float).eps
    )

    log_score = (
        -1.05 * normalized_length
        -0.20 * absolute_length
        +2.15 * np.log(reciprocal_rank)
        -0.90 * scale_difference
    )

    log_score -= np.max(log_score[valid])
    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
