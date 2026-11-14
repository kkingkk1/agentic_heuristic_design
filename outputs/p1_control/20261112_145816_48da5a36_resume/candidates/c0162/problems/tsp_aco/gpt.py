import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0.0)
    d = np.where(finite_positive, d, 0.0)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, global_scale)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    sorted_work = np.sort(work, axis=1)
    neighbor_count = min(2, n - 1)
    local_scale = np.mean(
        sorted_work[:, :neighbor_count],
        axis=1,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / geometric_scale
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    density = global_scale / local_scale
    log_density = np.log(np.maximum(density, eps))
    density_gap = np.abs(
        log_density[:, None] - log_density[None, :]
    )

    density_floor = 0.30 / (1.0 + 0.50 * density_gap)
    cross_density_factor = density_floor + (
        1.0 - density_floor
    ) * np.exp(-0.25 * density_gap)

    crowding = np.sum(
        np.exp(-np.minimum(
            work / local_scale[:, None],
            50.0,
        )) * valid,
        axis=1,
    )
    crowding_gap = np.abs(
        np.log1p(crowding[:, None])
        - np.log1p(crowding[None, :])
    )

    log_score = (
        -0.90 * relative_length
        -0.18 * global_length
        +1.65 * np.log(np.maximum(reciprocal_rank, eps))
        -0.45 * density_gap
        -0.20 * crowding_gap
        +np.log(np.maximum(cross_density_factor, eps))
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
