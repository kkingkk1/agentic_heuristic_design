import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    valid &= ~diagonal

    positive = valid & (distance_matrix > 0.0)
    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    distances = np.where(valid, distance_matrix, fallback)
    distances = np.maximum(distances, np.finfo(float).eps)
    distances[diagonal] = np.inf

    positive_rows = np.where(
        np.any(np.isfinite(distances) & ~diagonal, axis=1),
        1.0,
        fallback,
    )

    row_values = np.where(np.isfinite(distances), distances, fallback)
    row_values[diagonal] = np.nan
    row_scale = np.nanmedian(row_values, axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        positive_rows,
    )

    local_preference = 1.0 / (1.0 + distances / row_scale[:, None])
    local_preference[~np.isfinite(local_preference)] = 0.0
    local_preference[diagonal] = 0.0

    k = min(n - 1, max(2, int(np.ceil(np.sqrt(n)))))
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

    elite = np.zeros((n, n), dtype=float)
    elite[np.arange(n)[:, None], nearest] = 1.0
    elite[diagonal] = 0.0

    reciprocal_elite = np.sqrt(elite * elite.T)
    reciprocal_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    global_distance = 0.5 * (distances + distances.T)
    global_distance[diagonal] = np.inf
    global_preference = 1.0 / (1.0 + global_distance / max(float(np.nanmedian(global_distance)), np.finfo(float).eps))
    global_preference[~np.isfinite(global_preference)] = 0.0
    global_preference[diagonal] = 0.0

    score = reciprocal_preference * (
        0.65 * np.sqrt(reciprocal_elite)
        + 0.35
    )
    score *= np.sqrt(np.maximum(global_preference, 0.0))
    score = 0.5 * (score + score.T)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
