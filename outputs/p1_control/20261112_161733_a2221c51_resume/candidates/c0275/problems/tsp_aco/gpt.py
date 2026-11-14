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
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d = np.where(valid, symmetric_d, np.inf)

    positive_values = symmetric_d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    # Use the second available positive neighbor when possible.
    sorted_distances = np.sort(symmetric_d, axis=1)
    first_neighbor = sorted_distances[:, 0]
    second_neighbor = sorted_distances[:, 1]

    local_scale = np.where(
        np.isfinite(second_neighbor),
        second_neighbor,
        np.where(
            np.isfinite(first_neighbor),
            first_neighbor,
            global_scale,
        ),
    )
    local_scale = np.maximum(local_scale, eps)

    work = np.where(
        valid,
        symmetric_d,
        max(12.0 * fallback_distance, global_scale),
    )

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / np.maximum(endpoint_scale, eps)
    relative_length = np.minimum(relative_length, 1e6)

    global_length = work / global_scale
    global_length = np.minimum(global_length, 1e6)

    # Geometric endpoint potential favors short edges relative to both endpoints.
    endpoint_potential = np.sqrt(
        (local_scale[:, None] / np.maximum(work, eps))
        * (local_scale[None, :] / np.maximum(work, eps))
    )
    endpoint_potential = np.maximum(endpoint_potential, eps)

    # Stable bilateral ranking measures independent preference from both endpoints.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    valid_degree = valid.sum(axis=1).astype(float)
    rank_denominator = np.maximum(valid_degree - 1.0, 1.0)
    normalized_ranks = ranks / rank_denominator[:, None]
    normalized_ranks = np.minimum(normalized_ranks, 1.0)

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - normalized_ranks, 0.0)
        * np.maximum(1.0 - normalized_ranks.T, 0.0)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    # Comparable endpoint neighborhoods receive a mild compatibility bonus.
    log_scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.50 * np.minimum(log_scale_gap, 50.0)
    )
    scale_compatibility = np.maximum(scale_compatibility, eps)

    # Penalize edges that are long for at least one endpoint.
    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)
    excess_longness = np.minimum(excess_longness, 1e6)

    # Mildly preserve connectivity in sparse neighborhoods.
    degree_fraction = valid_degree / max(n - 1, 1)
    sparsity_bonus = 1.0 / np.sqrt(
        np.maximum(degree_fraction[:, None], eps)
        * np.maximum(degree_fraction[None, :], eps)
    )
    sparsity_bonus = np.minimum(sparsity_bonus, 8.0)

    log_score = (
        0.95 * np.log(endpoint_potential)
        + 1.35 * np.log(bilateral_rank)
        + 0.40 * np.log(scale_compatibility)
        - 0.58 * relative_length
        - 0.15 * global_length
        - 0.30 * excess_longness
        + 0.08 * np.log(sparsity_bonus)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

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
