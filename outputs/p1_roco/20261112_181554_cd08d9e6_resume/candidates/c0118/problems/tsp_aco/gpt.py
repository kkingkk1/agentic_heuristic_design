import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0.0)
    finite &= ~np.eye(n, dtype=bool)

    if not np.any(finite):
        return result

    # Work with a symmetric matrix for undirected TSP instances.
    d = np.where(finite, distances, np.inf)
    d = np.minimum(d, d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    if not np.any(valid):
        return result

    finite_values = d[valid]
    positive = finite_values[finite_values > 0.0]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = float(np.max(finite_values))

    global_scale = max(global_scale, np.finfo(float).eps)
    maximum_distance = max(float(np.max(finite_values)), global_scale)

    # Missing edges remain available to the sampler but receive a soft penalty.
    missing_penalty = 4.0 * maximum_distance + global_scale
    work = np.where(valid, d, missing_penalty)

    # Exclude the diagonal from all local-neighborhood calculations.
    np.fill_diagonal(work, missing_penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    # Use several local neighbors so the scale is less sensitive to one outlier.
    neighborhood_width = min(max(2, int(np.ceil(np.sqrt(n))), 4), n)
    local_neighbors = sorted_work[:, :neighborhood_width]

    positive_local = np.where(
        local_neighbors > 0.0,
        local_neighbors,
        global_scale,
    )
    local_scale = np.median(positive_local, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Robust relative distance, symmetric in the two endpoint scales.
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = work / pair_scale
    relative = np.maximum(relative, 0.0)

    distance_term = np.exp(-np.minimum(relative, 40.0))

    # Stable ranks from both endpoints.
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    rank_term = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    # Endpoint accessibility: favor edges that are locally competitive
    # from both directions.
    row_access = np.exp(
        -np.minimum(work / local_scale[:, None], 40.0)
    )
    col_access = np.exp(
        -np.minimum(work / local_scale[None, :], 40.0)
    )
    accessibility = np.sqrt(row_access * col_access)

    # Candidate scarcity: reward edges that are among the stronger options
    # of endpoints with few comparable alternatives.
    best = sorted_work[:, 0]
    second = sorted_work[:, 1] if n > 1 else sorted_work[:, 0]
    third = sorted_work[:, 2] if n > 2 else second

    eps = np.finfo(float).eps
    first_gap = np.maximum(second - best, 0.0) / np.maximum(second, eps)
    second_gap = np.maximum(third - second, 0.0) / np.maximum(third, eps)

    scarcity = np.clip(
        0.5 * first_gap + 0.5 * second_gap,
        0.0,
        1.0,
    )
    scarcity_bonus = 1.0 + 0.30 * (
        scarcity[:, None] + scarcity[None, :]
    )

    # Mutual nearest-neighbor edges receive a moderate structural bonus.
    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and j != i and nearest[j] == i:
            mutual[i, j] = True
    mutual_bonus = np.where(mutual, 1.22, 1.0)

    # Candidate-set overlap rewards edges supported by both endpoints'
    # local neighborhoods while retaining nonzero exploration probability.
    candidate_count = max(2, min(neighborhood_width, n - 1))
    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[
        np.arange(n)[:, None],
        order[:, :candidate_count],
    ] = True
    shared_candidate = candidate_mask & candidate_mask.T
    overlap_bonus = np.where(shared_candidate, 1.12, 1.0)

    # Smoothly suppress edges far beyond both endpoint neighborhoods.
    outlier_excess = np.maximum(relative - 2.0, 0.0)
    outlier_penalty = 1.0 / (1.0 + 0.75 * outlier_excess)

    # Mild global-scale correction prevents zero-distance duplicates from
    # dominating every other signal.
    scale_correction = 1.0 / (
        1.0 + 0.15 * np.maximum(work / global_scale - 1.0, 0.0)
    )

    scores = (
        distance_term
        * (0.42 + 0.23 * rank_term + 0.23 * accessibility + 0.12 * scale_correction)
        * scarcity_bonus
        * mutual_bonus
        * overlap_bonus
        * outlier_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
