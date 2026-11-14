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

    positive = distances[valid]
    if positive.size == 0:
        return scores

    eps = np.finfo(float).eps

    global_scale = np.median(positive)
    inverse_distance = np.zeros_like(distances, dtype=float)
    inverse_distance[valid] = 1.0 / (distances[valid] + eps)

    local_closeness = np.zeros_like(distances, dtype=float)
    nearest_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if np.any(row_valid):
            row_distances = distances[i, row_valid]
            nearest_scale[i] = np.median(row_distances)

    local_closeness[valid] = np.exp(
        -distances[valid] / (nearest_scale[:, None] + eps)[valid]
    )

    mutual_neighborhood = np.sqrt(
        np.maximum(local_closeness, 0.0)
        * np.maximum(local_closeness.T, 0.0)
    )

    def normalize(matrix: np.ndarray) -> np.ndarray:
        result = np.zeros_like(matrix, dtype=float)
        row_max = np.max(matrix, axis=1, keepdims=True)
        nonzero = row_max > 0
        result[nonzero[:, 0]] = (
            matrix[nonzero[:, 0]] / (row_max[nonzero[:, 0]] + eps)
        )
        return result

    global_component = normalize(inverse_distance)
    local_component = normalize(mutual_neighborhood)

    scores = 0.65 * global_component + 0.35 * local_component
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
