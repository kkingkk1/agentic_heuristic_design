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
    eps = np.finfo(float).eps
    fallback = float(np.max(positive_values))
    replacement = max(fallback * 10.0, eps)

    valid = finite_positive & finite_positive.T
    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    kth = min(1, n - 2)
    local_scale = np.maximum(
        np.partition(work, kth, axis=1)[:, kth],
        eps,
    )

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / endpoint_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    global_scale = max(float(np.median(positive_values)), eps)
    absolute_length = work / global_scale

    scale_sum = local_scale[:, None] + local_scale[None, :] + eps
    scale_difference = (
        np.abs(local_scale[:, None] - local_scale[None, :])
        / scale_sum
    )

    log_score = (
        -0.9 * normalized_length
        -0.35 * absolute_length
        + 1.8 * np.log(reciprocal_rank)
        -1.2 * scale_difference
    )

    valid_scores = log_score[valid]
    log_score -= np.max(valid_scores)

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))

    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
