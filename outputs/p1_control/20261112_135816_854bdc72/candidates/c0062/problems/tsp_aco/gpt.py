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
    if not np.all(finite):
        finite_values = distances[finite]
        replacement = (
            float(np.max(finite_values[finite_values >= 0]))
            if np.any(finite_values >= 0)
            else 1.0
        )
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse_distance = 1.0 / np.maximum(distances, np.finfo(float).eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_distance /= max(
        float(np.max(inverse_distance)),
        np.finfo(float).eps,
    )

    row_sorted = np.sort(distances, axis=1)
    local_scale = row_sorted[:, 0]
    fallback = np.median(positive) if positive.size else scale
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_proximity = np.exp(
        -0.85 * distances / np.sqrt(np.outer(local_scale, local_scale))
    )
    relative_proximity[~np.isfinite(relative_proximity)] = 0.0
    np.fill_diagonal(relative_proximity, 0.0)

    nearest_rank = np.argsort(np.argsort(distances, axis=1), axis=1)
    rank_score = 1.0 / (1.0 + nearest_rank.astype(float))
    rank_score /= np.max(rank_score, axis=1, keepdims=True)
    mutual_preference = np.sqrt(rank_score * rank_score.T)

    score = (
        0.55 * inverse_distance
        + 0.25 * relative_proximity
        + 0.20 * mutual_preference
    )
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
