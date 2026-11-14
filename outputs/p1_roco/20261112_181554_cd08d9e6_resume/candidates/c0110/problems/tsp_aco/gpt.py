import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    # Convert the instance to an undirected working matrix.
    directed = np.where(valid, distances, np.inf)
    d = 0.5 * (directed + directed.T)
    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)
    np.fill_diagonal(d, np.inf)

    if not np.any(valid):
        return scores

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.max(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    max_distance = max(float(np.max(finite_values)), global_scale)
    penalty = max_distance * 4.0 + global_scale

    work = np.where(valid, d, penalty)
    neighbor_work = work.copy()
    np.fill_diagonal(neighbor_work, np.inf)

    order = np.argsort(neighbor_work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(neighbor_work, order, axis=1)

    nearest = sorted_distances[:, 0]
    if n > 2:
        second_nearest = sorted_distances[:, 1]
        second_nearest = np.where(
            np.isfinite(second_nearest),
            second_nearest,
            nearest,
        )
    else:
        second_nearest = nearest.copy()

    # Robust local scale for each endpoint.
    local_scale = 0.5 * (nearest + second_nearest)
    local_scale = np.maximum(local_scale, global_scale * 1e-6)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    geometric_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    normalized_distance = work / geometric_scale

    # Smooth local-distance preference.
    distance_term = 1.0 / (1.0 + normalized_distance)

    # Candidate-rank preference, with a wider tail for larger instances.
    rank_temperature = max(2.0, np.sqrt(float(n)))
    rank_term = np.exp(-np.minimum(ranks, 60.0) / rank_temperature)

    # Each endpoint should regard the edge as locally accessible.
    endpoint_ratio_i = work / np.maximum(local_scale[:, None], eps)
    endpoint_ratio_j = work / np.maximum(local_scale[None, :], eps)
    accessibility = np.exp(
        -0.5 * np.minimum(endpoint_ratio_i + endpoint_ratio_j, 120.0)
    )

    # Prefer edges that are relatively scarce at one or both endpoints.
    regret = (second_nearest - nearest) / np.maximum(second_nearest, eps)
    regret = np.clip(regret, 0.0, 1.0)
    scarcity_bonus = 1.0 + 0.45 * (
        regret[:, None] + regret[None, :]
    )

    # Give a modest bonus to mutual nearest-neighbor edges.
    nearest_neighbor = order[:, 0]
    mutual = (
        (nearest_neighbor[:, None] == np.arange(n)[None, :])
        & (nearest_neighbor[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual, 1.18, 1.0)

    # Suppress edges that are extreme relative to both endpoint neighborhoods.
    excess = np.maximum(normalized_distance - 2.0, 0.0)
    outlier_penalty = 1.0 / (1.0 + excess)

    # Keep a positive exploratory component while emphasizing local candidates.
    base_score = (
        0.50 * distance_term
        + 0.25 * rank_term
        + 0.25 * accessibility
    )

    scores = (
        base_score
        * scarcity_bonus
        * mutual_bonus
        * outlier_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
