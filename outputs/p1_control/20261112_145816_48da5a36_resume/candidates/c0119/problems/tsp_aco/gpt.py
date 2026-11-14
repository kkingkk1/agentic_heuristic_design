import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    finite_values = distances[finite]

    if finite_values.size:
        nonnegative = finite_values[finite_values >= 0]
        replacement = (
            float(np.max(nonnegative))
            if nonnegative.size
            else 1.0
        )
    else:
        replacement = 1.0

    distances = np.where(finite, distances, replacement)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    positive = distances[distances > 0]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    working = distances.copy()
    np.fill_diagonal(working, np.inf)

    sorted_distances = np.sort(working, axis=1)
    finite_sorted = np.where(np.isfinite(sorted_distances), sorted_distances, global_scale)

    k = max(1, min(4, n - 1))
    local_scale = np.mean(finite_sorted[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(np.outer(local_scale, local_scale))
    normalized_distance = distances / np.maximum(pair_scale, eps)

    proximity = np.exp(-0.75 * normalized_distance)
    proximity[~np.isfinite(proximity)] = 0.0
    np.fill_diagonal(proximity, 0.0)

    inverse_distance = 1.0 / np.maximum(distances, eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_max = float(np.max(inverse_distance))
    if inverse_max > 0:
        inverse_distance /= inverse_max

    ranks = np.argsort(np.argsort(working, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )
    mutual_preference = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )
    np.fill_diagonal(mutual_preference, 0.0)

    local_competition = np.exp(
        -0.5 * normalized_distance
    )
    local_competition[~np.isfinite(local_competition)] = 0.0
    np.fill_diagonal(local_competition, 0.0)

    score = (
        0.50 * inverse_distance
        + 0.25 * proximity
        + 0.15 * mutual_preference
        + 0.10 * local_competition
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
