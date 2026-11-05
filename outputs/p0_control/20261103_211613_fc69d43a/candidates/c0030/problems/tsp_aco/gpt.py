import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    safe_distances = np.where(valid, distances, np.inf)

    nearest = np.min(safe_distances, axis=1)
    second_nearest = np.partition(safe_distances, 1, axis=1)[:, 1] if n > 1 else nearest.copy()

    nearest = np.where(np.isfinite(nearest), nearest, 1.0)
    second_nearest = np.where(np.isfinite(second_nearest), second_nearest, nearest)

    endpoint_pressure = second_nearest / np.maximum(nearest, np.finfo(float).eps)
    endpoint_pressure = np.sqrt(endpoint_pressure[:, None] * endpoint_pressure[None, :])

    scale = np.median(distances[valid]) if np.any(valid) else 1.0
    scale = max(scale, np.finfo(float).eps)

    normalized = distances / scale
    direct = 1.0 / (normalized + np.finfo(float).eps)

    detour_factor = 1.0 + np.log1p(np.maximum(endpoint_pressure, 0.0))
    scores = direct * detour_factor

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores
