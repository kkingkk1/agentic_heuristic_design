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

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]
    fallback = max(float(np.max(positive_values)), eps)
    replacement = max(fallback * 10.0, eps)

    valid = finite_positive & finite_positive.T
    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    sorted_distances = np.sort(work, axis=1)
    neighbor_count = max(1, min(2, n - 1))
    local_scale = np.mean(
        sorted_distances[:, :neighbor_count],
        axis=1,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / geometric_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    global_scale = max(float(np.median(positive_values)), eps)
    absolute_length = work / global_scale

    density = global_scale / local_scale
    density_log_gap = np.abs(
        np.log(np.maximum(density[:, None], eps))
        - np.log(np.maximum(density[None, :], eps))
    )
    density_log_gap /= 1.0 + np.abs(
        np.log(np.maximum(density[:, None], eps))
        + np.log(np.maximum(density[None, :], eps))
    )

    cross_density_floor = 0.35 / (
        1.0 + 0.5 * density_log_gap
    )

    log_score = (
        -0.72 * normalized_length
        -0.22 * absolute_length
        + 1.9 * np.log(reciprocal_rank)
        - 0.55 * density_log_gap
    )

    raw_score = np.exp(np.clip(log_score, -745.0, 0.0))
    score = raw_score * (
        cross_density_floor
        + (1.0 - cross_density_floor)
        * np.exp(-0.35 * normalized_length)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
