import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)

    distances = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0)

    if np.any(positive):
        scale = float(np.median(distances[positive]))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)
    stabilizer = scale * 1e-7

    inverse_distance = np.zeros_like(distances)
    inverse_distance[finite] = 1.0 / (distances[finite] + stabilizer)

    nearest = np.min(
        np.where(finite, distances, np.inf),
        axis=1,
    )
    nearest = np.where(np.isfinite(nearest), nearest, scale)

    denominator = nearest[:, None] + nearest[None, :] + stabilizer
    local_support = np.exp(-distances / denominator)
    local_support[~finite] = 0.0

    nearest_indices = np.argmin(distances, axis=1)
    mutual_nearest = np.zeros_like(distances)

    for i in range(n):
        j = nearest_indices[i]
        if j != i and nearest_indices[j] == i:
            mutual_nearest[i, j] = 1.0

    score = inverse_distance * (0.65 + 0.35 * local_support)
    score *= 1.0 + 0.30 * mutual_nearest

    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
