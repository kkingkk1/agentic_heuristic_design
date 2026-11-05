import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive edge scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0)

    if not np.any(positive):
        return np.zeros_like(distances, dtype=float)

    valid_values = distances[positive]
    scale = np.median(valid_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(valid_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_distances = np.where(positive, distances, np.inf)

    global_term = 1.0 / (1.0 + safe_distances / scale)

    row_min = np.min(safe_distances, axis=1)
    row_min = np.where(np.isfinite(row_min), row_min, scale)

    row_mean = np.mean(
        np.where(np.isfinite(safe_distances), safe_distances, row_min[:, None]),
        axis=1,
    )
    row_mean = np.maximum(row_mean, np.finfo(float).eps)

    local_term = 0.5 * (
        1.0 / (1.0 + safe_distances / row_mean[:, None])
        + 1.0 / (1.0 + safe_distances / row_mean[None, :])
    )

    score = global_term * (0.5 + local_term)

    score[~positive] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
