import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    with np.errstate(invalid="ignore", over="ignore"):
        symmetric_d = 0.5 * (d + d.T)

    edge_values = symmetric_d[valid]
    global_scale = max(float(np.median(edge_values)), eps)

    work = np.where(valid, symmetric_d, np.inf)
    degree = valid.sum(axis=1)

    nearest_scale = np.min(work, axis=1)
    second_scale = np.partition(work, 1, axis=1)[:, 1]

    local_scale = np.where(degree >= 2, second_scale, nearest_scale)
    local_scale = np.where(
        degree > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_length = work / endpoint_scale
        global_length = work / global_scale

        row_ratio = work / local_scale[:, None]
        col_ratio = work / local_scale[None, :]

        bilateral_log_preference = -0.5 * (
            np.log1p(row_ratio) + np.log1p(col_ratio)
        )

        scale_mismatch = np.abs(
            np.log(local_scale[:, None])
            - np.log(local_scale[None, :])
        )
        compatibility_penalty = -0.12 * np.minimum(scale_mismatch, 20.0)

    nearest = np.argmin(work, axis=1)
    indices = np.arange(n)

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )

    local_radius = 1.5 * local_scale[:, None]
    nearby_count = np.sum(
        valid & (work <= local_radius),
        axis=1,
    )
    local_density = nearby_count / np.maximum(degree, 1)
    excess_density = np.maximum(local_density - 0.5, 0.0)

    congestion = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            -0.72 * np.log(np.maximum(global_length, eps))
            + 1.05 * bilateral_log_preference
            - 0.55 * np.minimum(relative_length, 6.0)
            + 0.18 * mutual_nearest.astype(float)
            + compatibility_penalty
            - 0.10 * congestion
        )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))

    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(np.clip(log_score - shift, -745.0, 0.0))

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
