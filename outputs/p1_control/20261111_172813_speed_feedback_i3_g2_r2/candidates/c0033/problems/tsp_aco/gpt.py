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

    original = distance_matrix.copy()
    diagonal = np.eye(n, dtype=bool)

    valid = np.isfinite(original) & (original >= 0.0) & ~diagonal
    positive = valid & (original > 0.0)

    if np.any(positive):
        fallback = float(np.median(original[positive]))
        fallback = max(fallback, np.finfo(float).eps)
    else:
        fallback = 1.0

    distances = original.copy()
    distances[~valid] = fallback
    distances = np.maximum(distances, np.finfo(float).eps)
    distances[diagonal] = np.inf

    inverse_distance = 1.0 / distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0

    # Slightly soften extreme nearest-neighbor dominance.
    local_preference = np.sqrt(inverse_distance / row_max)

    order = np.argsort(-local_preference, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values

    # Logarithmic decay retains nonzero exploration for lower-ranked edges.
    rank_weight = 1.0 / np.log2(ranks + 1.0) ** 0.85
    rank_weight[diagonal] = 0.0

    endpoint_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )
    mutual_rank_weight = np.sqrt(rank_weight * rank_weight.T)

    score = endpoint_preference * mutual_rank_weight
    score = 0.5 * (score + score.T)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
