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

    if not finite_positive.any():
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive_values)), eps)

    maximum_distance = float(np.max(positive_values))
    replacement = min(
        maximum_distance * 10.0,
        np.finfo(float).max,
    )
    replacement = max(replacement, eps)

    valid = finite_positive
    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    nearest = np.partition(
        work,
        neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]
    local_scale = np.maximum(nearest.mean(axis=1), eps)

    local_scale_product = local_scale[:, None] * local_scale[None, :]
    normalized_length = work / np.sqrt(local_scale_product)

    reciprocal_rank = _reciprocal_rank(work)

    log_density = np.log(global_scale / local_scale)
    density_difference = log_density[:, None] - log_density[None, :]
    density_gap = np.abs(density_difference)
    density_gap /= 1.0 + np.abs(
        log_density[:, None] + log_density[None, :]
    )

    cross_density_floor = 0.35 / (1.0 + 0.5 * density_gap)

    log_score = (
        -0.72 * normalized_length
        -0.22 * (work / global_scale)
        + 1.9 * np.log(reciprocal_rank)
        - 0.55 * density_gap
    )

    raw_score = np.exp(np.clip(log_score, -745.0, 0.0))
    score = raw_score * (
        cross_density_floor
        + (1.0 - cross_density_floor)
        * np.exp(-0.35 * normalized_length)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = (score + score.T) * 0.5
    maximum_score = float(np.max(score))

    if maximum_score > 0.0:
        score /= maximum_score

    return score.astype(float, copy=False)


def _reciprocal_rank(work: np.ndarray) -> np.ndarray:
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(work, dtype=float)

    rows = np.arange(work.shape[0])[:, None]
    ranks[rows, order] = np.arange(work.shape[1], dtype=float)

    return 1.0 / (1.0 + ranks + ranks.T)
