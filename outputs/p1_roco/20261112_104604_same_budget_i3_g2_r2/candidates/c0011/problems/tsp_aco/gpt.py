import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    distances = distance_matrix.copy()
    finite = np.isfinite(distances)

    if not finite.any():
        return np.zeros_like(distances, dtype=float)

    finite_positive = finite & (distances > 0)
    positive_values = distances[finite_positive]

    if positive_values.size == 0:
        result = np.zeros_like(distances, dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    large_value = max(float(np.max(positive_values)) * 10.0, 1.0)
    distances[~finite] = large_value
    distances[distances <= 0] = large_value
    np.fill_diagonal(distances, large_value)

    eps = np.finfo(float).eps

    inverse = 1.0 / (distances + eps)

    row_min = np.min(distances, axis=1, keepdims=True)
    row_max = np.max(distances, axis=1, keepdims=True)
    closeness = (row_max - distances) / (row_max - row_min + eps)

    nearest = np.min(distances, axis=1, keepdims=True)
    mutual_preference = (
        (nearest / (distances + eps))
        * (nearest.T / (distances.T + eps))
    )

    if n > 2:
        sorted_distances = np.sort(distances, axis=1)
        second_nearest = sorted_distances[:, 1:2]
        savings = (second_nearest - distances) / (
            second_nearest + distances + eps
        )
        savings = np.maximum(savings, 0.0)
    else:
        savings = np.zeros_like(distances)

    def row_normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.maximum(matrix, 0.0)
        scale = np.max(matrix, axis=1, keepdims=True)
        return matrix / (scale + eps)

    score = (
        0.42 * row_normalize(inverse)
        + 0.28 * row_normalize(closeness)
        + 0.25 * row_normalize(mutual_preference)
        + 0.05 * row_normalize(savings)
    )

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.maximum(score, 0.0)
