import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite = np.isfinite(d)
    nonnegative = finite & (d >= 0.0)
    valid = nonnegative & nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d = np.maximum(symmetric_d, 0.0)

    positive_values = symmetric_d[valid & (symmetric_d > 0.0)]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    maximum_distance = (
        float(np.max(symmetric_d[valid]))
        if np.any(valid)
        else global_scale
    )
    replacement = max(10.0 * maximum_distance, global_scale)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # Use the second-nearest valid edge as the local scale when possible.
    neighbor_count = min(2, max(n - 1, 1))
    nearest = np.partition(work, neighbor_count - 1, axis=1)
    local_scale = nearest[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / endpoint_scale
    global_length = work / global_scale

    # Soft acceptance from each endpoint's local perspective.
    endpoint_acceptance = 1.0 / (
        1.0 + np.maximum(work / local_scale[:, None] - 1.0, 0.0)
    )
    endpoint_acceptance *= 1.0 / (
        1.0 + np.maximum(work / local_scale[None, :] - 1.0, 0.0)
    )
    endpoint_acceptance = np.sqrt(
        np.maximum(endpoint_acceptance, eps)
    )

    # Continuous bilateral rank preference.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = float(n)

    rank_fraction = ranks / max(n - 1, 1)
    bilateral_rank = np.exp(
        -0.5 * (rank_fraction + rank_fraction.T)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    # Prefer endpoints with compatible local neighborhood scales.
    scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.30 * np.minimum(scale_gap, 40.0)
    )
    scale_compatibility = np.maximum(
        scale_compatibility,
        eps,
    )

    # Estimate local crowding and penalize only long edges in crowded regions.
    density_radius = 1.35 * local_scale
    local_density = np.mean(
        valid & (work <= density_radius[:, None]),
        axis=1,
    )
    excess_density = np.maximum(local_density - 0.40, 0.0)
    crowding_penalty = (
        excess_density[:, None]
        + excess_density[None, :]
    ) * np.minimum(relative_length, 4.0)

    reciprocal_global = global_scale / np.maximum(work, eps)
    reciprocal_global = np.minimum(reciprocal_global, 1e12)

    log_score = (
        1.05 * np.log(np.maximum(endpoint_acceptance, eps))
        + 0.95 * np.log(bilateral_rank)
        + 0.40 * np.log(scale_compatibility)
        + 0.35 * np.log(np.maximum(reciprocal_global, eps))
        - 0.32 * np.minimum(relative_length, 20.0)
        - 0.10 * np.minimum(global_length, 50.0)
        - 0.12 * crowding_penalty
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum_log_score = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(
            log_score[finite_scores] - maximum_log_score,
            -745.0,
            0.0,
        )
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum_score = float(np.max(score))
    if maximum_score > 0.0:
        score /= maximum_score

    return score.astype(float, copy=False)
