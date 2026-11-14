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
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, eps)

    valid = finite_positive & finite_positive.T

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    nearest = np.partition(work, neighbor_count - 1, axis=1)[:, :neighbor_count]
    local_scale = np.maximum(nearest.mean(axis=1), eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / geometric_scale
    absolute_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)
    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    log_density = np.log(global_scale / local_scale)
    density_log_gap = np.abs(
        log_density[:, None] - log_density[None, :]
    )
    density_log_gap /= (
        1.0
        + np.abs(
            log_density[:, None] + log_density[None, :]
        )
    )

    cross_density_floor = 0.35 / (1.0 + 0.5 * density_log_gap)

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
