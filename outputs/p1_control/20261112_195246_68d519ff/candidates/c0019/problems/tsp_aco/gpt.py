import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distance_matrix)
    positive = finite & (distance_matrix > 0)

    valid_values = distance_matrix[positive]
    if valid_values.size == 0:
        return scores

    fallback = float(np.max(valid_values))
    distances = np.where(positive, distance_matrix, fallback)

    np.fill_diagonal(distances, fallback)

    row_scale = np.median(np.where(positive, distances, np.nan), axis=1)
    row_scale = np.where(np.isfinite(row_scale) & (row_scale > 0), row_scale, fallback)

    normalized = distances / (row_scale[:, None] + 1e-12)

    local_preference = np.exp(-np.clip(normalized, 0.0, 50.0))

    row_mass = np.sum(local_preference, axis=1, keepdims=True) - np.diag(
        local_preference
    )[:, None]
    row_mass = np.maximum(row_mass, 1e-12)

    competition = local_preference / row_mass
    endpoint_support = np.sqrt(competition * competition.T)

    reciprocal_closeness = 1.0 / (1.0 + normalized)
    reciprocal_closeness = np.sqrt(
        reciprocal_closeness * reciprocal_closeness.T
    )

    scores = endpoint_support * reciprocal_closeness
    scores *= finite & ~np.eye(n, dtype=bool)

    if np.any(scores > 0):
        scores /= np.max(scores)
        scores = 0.05 + 0.95 * scores

    np.fill_diagonal(scores, 0.0)
    return scores.astype(float, copy=False)
