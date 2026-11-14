import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    distances = np.array(distance_matrix, dtype=float, copy=True)
    finite_values = distances[np.isfinite(distances) & (distances >= 0)]

    if finite_values.size == 0:
        return np.zeros_like(distances, dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    distances[~np.isfinite(distances)] = fallback
    distances[distances < 0] = fallback
    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    inverse_distance = np.zeros_like(distances, dtype=float)
    valid = np.isfinite(distances)
    inverse_distance[valid] = 1.0 / (distances[valid] + 1e-12)
    inverse_distance[~valid] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    local_preference = inverse_distance / row_max

    order = np.argsort(distances, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values[None, :]
    rank_score = 1.0 - ranks / max(n - 1, 1)
    rank_score[~valid] = 0.0

    reciprocal_score = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    distance_score = np.exp(-distances / scale)
    distance_score[~valid] = 0.0

    scores = (
        0.45 * distance_score
        + 0.30 * reciprocal_score
        + 0.25 * rank_score
    )

    scores = np.maximum(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
