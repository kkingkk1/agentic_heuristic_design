import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be a square 2D array")

    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distance_matrix)
    valid = finite & off_diagonal

    positive = valid & (distance_matrix > 0)
    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    distances = np.where(finite, distance_matrix, fallback)
    distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(distances, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / distances[valid]

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    preference = inverse / row_max
    preference[~valid] = 0.0

    k = min(n - 1, max(2, int(np.ceil(np.log2(n)))))

    masked = np.where(valid, preference, -np.inf)
    kth_values = np.partition(
        masked,
        kth=n - k,
        axis=1,
    )[:, n - k][:, None]

    candidate = valid & (preference >= kth_values)
    candidate = candidate | candidate.T

    score = 0.85 * preference + 0.15 * candidate.astype(float)
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
