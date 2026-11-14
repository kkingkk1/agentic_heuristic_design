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
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]

    global_scale = max(float(np.median(positive_values)), eps)
    upper_scale = max(float(np.percentile(positive_values, 90)), global_scale)
    replacement = max(upper_scale * 4.0, eps)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    valid = finite_positive & finite_positive.T

    neighbor_count = min(2, n - 1)
    nearest = np.partition(work, neighbor_count - 1, axis=1)[:, :neighbor_count]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    local_geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / local_geometric_scale

    row_order = np.argsort(work, axis=1, kind="stable")
    row_ranks = np.empty((n, n), dtype=float)
    row_ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    reciprocal_rank = 1.0 / (
        1.0 + row_ranks + row_ranks.T
    )

    density = global_scale / local_scale
    log_density = np.log(np.maximum(density, eps))
    density_gap = np.abs(
        log_density[:, None] - log_density[None, :]
    )
    density_level = 0.5 * (
        log_density[:, None] + log_density[None, :]
    )

    congestion = np.maximum(density_level, 0.0)
    cross_density_floor = 0.28 + 0.20 * np.exp(-0.5 * density_gap)
    cross_density_floor /= 1.0 + 0.15 * congestion
    cross_density_floor = np.clip(
        cross_density_floor,
        0.18,
        0.48,
    )

    absolute_length = work / global_scale
    local_preference = np.exp(
        -np.minimum(normalized_length, 40.0)
    )
    density_adjustment = np.exp(
        -0.30 * np.minimum(congestion, 20.0)
        -0.18 * np.minimum(density_gap, 20.0)
    )

    log_score = (
        -0.82 * np.minimum(normalized_length, 80.0)
        -0.12 * np.minimum(absolute_length, 80.0)
        + 2.15 * np.log(np.maximum(reciprocal_rank, eps))
        + np.log(np.maximum(density_adjustment, eps))
    )

    base_score = np.exp(np.clip(log_score, -745.0, 0.0))
    score = base_score * (
        cross_density_floor
        + (1.0 - cross_density_floor) * local_preference
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
