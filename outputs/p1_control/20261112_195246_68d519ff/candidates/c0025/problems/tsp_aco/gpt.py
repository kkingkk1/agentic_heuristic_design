import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    positive = distances > 0
    valid = finite & positive
    np.fill_diagonal(valid, False)

    safe_distances = np.where(valid, distances, np.inf)

    inverse = np.zeros_like(distances, dtype=float)
    inverse[valid] = 1.0 / np.maximum(safe_distances[valid], np.finfo(float).eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    nearest_score = inverse / row_max

    nearest_indices = np.argmin(safe_distances, axis=1)
    mutual = np.zeros_like(distances, dtype=float)
    for i in range(n):
        j = nearest_indices[i]
        if j < n and nearest_indices[j] == i and valid[i, j]:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    local_scale = np.sum(np.where(valid, distances, 0.0), axis=1)
    local_count = np.sum(valid, axis=1)
    local_mean = local_scale / np.maximum(local_count, 1)
    accessibility = np.zeros_like(distances, dtype=float)

    for i in range(n):
        for j in range(n):
            if valid[i, j]:
                denominator = local_mean[i] + local_mean[j]
                if denominator > 0:
                    accessibility[i, j] = 2.0 * local_mean[i] * local_mean[j] / (
                        denominator * distances[i, j]
                    )

    accessibility_max = np.max(accessibility, axis=1, keepdims=True)
    accessibility_max[accessibility_max == 0] = 1.0
    accessibility /= accessibility_max

    score = 0.60 * nearest_score + 0.25 * mutual + 0.15 * accessibility
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score
