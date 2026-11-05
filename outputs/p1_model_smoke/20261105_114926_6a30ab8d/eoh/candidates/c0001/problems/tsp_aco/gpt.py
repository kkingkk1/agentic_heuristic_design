import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    if not np.any(finite_positive):
        return np.zeros_like(distances, dtype=float)

    fallback = np.max(distances[finite_positive])
    safe_distances = np.where(
        finite_positive,
        distances,
        fallback,
    )

    np.fill_diagonal(safe_distances, np.inf)

    inverse_distance = 1.0 / safe_distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_scale = np.mean(
        np.where(np.isfinite(safe_distances), safe_distances, fallback),
        axis=1,
    )
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    proximity = np.sqrt(
        (row_scale[:, None] / safe_distances)
        * (row_scale[None, :] / safe_distances)
    )
    proximity[~np.isfinite(proximity)] = 0.0

    scores = inverse_distance * (1.0 + proximity)

    scores[~finite_positive] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
