import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        global_scale = max(float(np.median(positive)), eps)
        replacement = max(float(np.max(positive)) * 10.0, global_scale)
    else:
        finite_values = d[valid]
        global_scale = (
            max(float(np.median(finite_values)), eps)
            if finite_values.size
            else 1.0
        )
        replacement = global_scale * 10.0

    work = np.where(valid, np.maximum(d, 0.0), replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    nearest_values = np.partition(
        work,
        neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    local_scale = np.maximum(
        np.mean(nearest_values, axis=1),
        eps,
    )

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_length = work / geometric_scale
    global_length = work / global_scale

    endpoint_quality = 1.0 / (
        1.0
        + work / np.maximum(local_scale[:, None], eps)
    )
    endpoint_quality_t = 1.0 / (
        1.0
        + work / np.maximum(local_scale[None, :], eps)
    )
    bilateral_quality = np.sqrt(
        np.maximum(endpoint_quality * endpoint_quality_t, eps)
    )

    reciprocal_distance = global_scale / np.maximum(work, eps)

    mutual_nearest = (
        work
        <= np.minimum(
            np.min(work, axis=1)[:, None],
            np.min(work, axis=1)[None, :],
        )
    ).astype(float)

    density_radius = 1.5 * local_scale[:, None]
    local_density = np.mean(
        valid & (work <= density_radius),
        axis=1,
    )

    excess_density = np.maximum(local_density - 0.35, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(normalized_length, 3.0)

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_mismatch = np.minimum(scale_mismatch, 20.0)

    log_score = (
        0.95 * np.log(np.maximum(reciprocal_distance, eps))
        + 1.10 * np.log(np.maximum(bilateral_quality, eps))
        - 0.85 * normalized_length
        - 0.22 * global_length
        - 0.12 * scale_mismatch
        - 0.08 * density_penalty
        + 0.20 * mutual_nearest
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))

    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
