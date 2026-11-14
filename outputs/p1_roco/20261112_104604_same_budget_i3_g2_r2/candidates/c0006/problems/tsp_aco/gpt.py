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

    if not np.any(valid):
        return scores

    safe_distances = np.where(valid, distances, np.inf)

    row_scale = np.full(n, np.inf, dtype=float)
    for i in range(n):
        row_values = distances[i, valid[i]]
        if row_values.size:
            row_scale[i] = np.median(row_values)

    inverse_distance = np.zeros_like(distances, dtype=float)
    inverse_distance[valid] = 1.0 / (distances[valid] + 1e-12)

    local_affinity = np.zeros_like(distances, dtype=float)
    for i in range(n):
        if np.isfinite(row_scale[i]):
            local_affinity[i, valid[i]] = np.exp(
                -distances[i, valid[i]] / (row_scale[i] + 1e-12)
            )

    neighborhood_similarity = np.zeros_like(distances, dtype=float)
    comparable = valid & np.isfinite(row_scale[:, None]) & np.isfinite(row_scale[None, :])
    neighborhood_similarity[comparable] = np.sqrt(
        local_affinity[comparable] *
        local_affinity.T[comparable]
    )

    nearest = np.min(safe_distances, axis=1)
    mutual_preference = np.zeros_like(distances, dtype=float)
    mutual_preference[valid] = (
        nearest[:, None][valid] / (distances[valid] + 1e-12)
        * nearest[None, :][valid] / (distances[valid] + 1e-12)
    )

    def normalize_component(component: np.ndarray) -> np.ndarray:
        component = np.maximum(component, 0.0)
        maximum = np.max(component)
        if maximum > 0:
            return component / maximum
        return component

    inverse_distance = normalize_component(inverse_distance)
    neighborhood_similarity = normalize_component(neighborhood_similarity)
    mutual_preference = normalize_component(mutual_preference)

    scores = (
        0.55 * inverse_distance
        + 0.25 * mutual_preference
        + 0.20 * neighborhood_similarity
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
