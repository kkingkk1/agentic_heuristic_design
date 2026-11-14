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

    finite = np.isfinite(d)
    positive = finite & (d > 0.0)
    valid = positive & positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    positive_values = symmetric_d[valid]

    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(10.0 * float(np.max(positive_values)), global_scale)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # The second-nearest valid edge provides a robust local scale.
    kth = min(1, n - 2)
    local_scale = np.partition(work, kth, axis=1)[:, kth]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / endpoint_scale
    global_length = work / global_scale

    # Continuous bilateral neighborhood preference.
    normalized_rows = work / local_scale[:, None]
    rank_surrogate = 1.0 / (1.0 + normalized_rows)
    bilateral_preference = np.sqrt(
        rank_surrogate * rank_surrogate.T
    )

    reciprocal_preference = global_scale / np.maximum(work, eps)
    reciprocal_preference = np.clip(reciprocal_preference, eps, 1e12)

    # Soft reward for edges that are among the nearest neighbors of both endpoints.
    nearest = np.argmin(work, axis=1)
    mutual_nearest = (
        nearest[:, None] == np.arange(n)[None, :]
    ) & (
        nearest[None, :] == np.arange(n)[:, None]
    )
    mutual_nearest = mutual_nearest & valid

    # Penalize incompatible endpoint neighborhoods mildly.
    scale_mismatch = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.35 * np.minimum(scale_mismatch, 20.0)
    )

    # Penalize only local over-congestion, preserving necessary long edges.
    local_radius = 1.5 * local_scale[:, None]
    local_density = np.mean(
        valid & (work <= local_radius),
        axis=1,
    )
    excess_density = np.maximum(local_density - 0.35, 0.0)
    congestion = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.85 * np.log(np.maximum(reciprocal_preference, eps))
            + 1.05 * np.log(np.maximum(bilateral_preference, eps))
            - 0.85 * relative_length
            - 0.12 * global_length
            + 0.20 * mutual_nearest.astype(float)
            + 0.40 * np.log(np.maximum(scale_compatibility, eps))
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
