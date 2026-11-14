import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    positive_distances = distances[valid]
    if positive_distances.size == 0:
        return scores

    scale = np.median(positive_distances)
    inverse_distance = np.zeros_like(distances, dtype=float)
    inverse_distance[valid] = 1.0 / (distances[valid] + 1e-12)
    inverse_distance /= np.max(inverse_distance)

    neighborhood = np.zeros_like(distances, dtype=float)
    for i in range(n):
        row = distances[i].copy()
        row[~valid[i]] = np.inf
        finite = np.isfinite(row)

        if np.any(finite):
            local_scale = np.median(row[finite])
            neighborhood[i, finite] = np.exp(-row[finite] / (local_scale + 1e-12))

    neighborhood_similarity = np.sqrt(
        np.maximum(neighborhood, 0.0) *
        np.maximum(neighborhood.T, 0.0)
    )

    scale_bonus = np.exp(
        -np.maximum(distances, 0.0) / (scale + 1e-12)
    )
    scale_bonus[~valid] = 0.0

    scores = (
        0.65 * inverse_distance
        + 0.25 * neighborhood_similarity
        + 0.10 * scale_bonus
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
