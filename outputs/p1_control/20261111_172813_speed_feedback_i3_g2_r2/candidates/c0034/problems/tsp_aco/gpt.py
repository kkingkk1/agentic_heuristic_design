import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0) & ~diagonal
    positive = valid & (distance_matrix > 0.0)

    fallback = float(np.median(distance_matrix[positive])) if np.any(positive) else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    distances = distance_matrix.copy()
    distances[~valid] = fallback
    distances = np.maximum(distances, np.finfo(float).eps)
    distances[diagonal] = np.inf

    inverse_distance = np.divide(
        1.0,
        distances,
        out=np.zeros_like(distances),
        where=np.isfinite(distances),
    )

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max = np.maximum(row_max, 1.0)
    local_preference = inverse_distance / row_max

    preference_power = 0.85
    local_preference = np.power(local_preference, preference_power)

    order = np.argsort(-local_preference, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values

    rank_base = 1.7
    rank_weight = 1.0 / np.log2(ranks + rank_base)
    rank_weight[diagonal] = 0.0

    endpoint_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )
    endpoint_rank = np.sqrt(rank_weight * rank_weight.T)

    score = endpoint_preference * endpoint_rank
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
