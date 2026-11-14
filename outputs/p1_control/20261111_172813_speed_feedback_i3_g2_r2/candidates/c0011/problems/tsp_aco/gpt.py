import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    distances = distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)

    finite = np.isfinite(distances)
    if not np.any(finite):
        return np.zeros_like(distances, dtype=float)

    positive = finite & (distances > 0)
    fallback = np.median(distances[positive]) if np.any(positive) else 1.0

    distances[~finite] = fallback
    distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(distances, np.inf)

    inverse_distance = 1.0 / distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_scale = np.max(inverse_distance, axis=1, keepdims=True)
    row_scale[row_scale <= 0.0] = 1.0
    local_preference = inverse_distance / row_scale

    concentration = np.exp(-5.0 * local_preference)
    competition = 1.0 / (
        np.sum(concentration, axis=1, keepdims=True) + 1e-12
    )

    maximum_competition = np.max(competition)
    if maximum_competition > 0.0:
        competition = competition / maximum_competition

    endpoint_factor = np.sqrt(competition @ competition.T)
    score = local_preference * endpoint_factor

    score = 0.5 * (score + score.T)
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score = score / maximum

    return score.reshape(distance_matrix.shape)
