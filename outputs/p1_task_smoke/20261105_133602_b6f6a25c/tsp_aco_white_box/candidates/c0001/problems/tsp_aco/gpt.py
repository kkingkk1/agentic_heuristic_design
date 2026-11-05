import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix must be nonnegative")

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    distances = distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)

    k = max(1, min(5, n - 1))
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = distances / endpoint_scale
    score = 1.0 / (1.0 + relative_distance)

    nearest_neighbor = np.argmin(distances, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_neighbor[i]
        if nearest_neighbor[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    score *= 1.0 + 0.25 * mutual_nearest
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score
