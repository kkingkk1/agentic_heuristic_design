import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = np.maximum(distances, 0.0)

    both_finite = finite & finite.T
    one_finite = finite ^ finite.T

    symmetric_distances = np.full((n, n), np.inf, dtype=float)
    symmetric_distances[both_finite] = (
        nonnegative[both_finite] + nonnegative.T[both_finite]
    ) / 2.0
    symmetric_distances[one_finite] = np.minimum(
        nonnegative[one_finite],
        nonnegative.T[one_finite],
    )
    np.fill_diagonal(symmetric_distances, np.inf)

    valid = np.isfinite(symmetric_distances)
    positive = valid & (symmetric_distances > 0)

    positive_values = symmetric_distances[positive]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    local_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = symmetric_distances[i, positive[i]]
        if values.size:
            local_scale[i] = max(float(np.median(values)), np.finfo(float).eps)

    safe_distance = np.where(valid, symmetric_distances, np.inf)
    epsilon = np.finfo(float).eps * global_scale

    inverse_distance = global_scale / (safe_distance + epsilon)
    inverse_distance[~valid] = 0.0
    maximum = float(np.max(inverse_distance))
    if maximum > 0:
        inverse_distance /= maximum

    relative_distance = safe_distance / np.sqrt(
        np.outer(local_scale, local_scale)
    )
    relative_preference = np.exp(-relative_distance)
    relative_preference[~valid] = 0.0

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[
                np.argsort(symmetric_distances[i, candidates], kind="stable")
            ]
            ranks[i, order] = np.arange(order.size, dtype=float)

    rank_preference = 1.0 / (1.0 + ranks)
    rank_preference[~valid] = 0.0
    mutual_preference = np.sqrt(rank_preference * rank_preference.T)

    nearest_k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[
                np.argsort(symmetric_distances[i, candidates], kind="stable")
            ]
            nearest[i, order[:nearest_k]] = 1.0

    mutual_neighbor_bonus = nearest * nearest.T

    score = (
        0.45 * inverse_distance
        + 0.25 * relative_preference
        + 0.20 * mutual_preference
        + 0.10 * mutual_neighbor_bonus
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
