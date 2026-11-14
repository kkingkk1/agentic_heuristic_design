import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distance_matrix)
    positive = finite & (distance_matrix > 0)

    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    distances = np.where(finite, distance_matrix, fallback)
    distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(distances, np.inf)

    inverse = np.zeros_like(distances, dtype=float)
    valid = np.isfinite(distances)
    inverse[valid] = 1.0 / distances[valid]

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    preference = inverse / row_max

    order = np.argsort(-preference, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = rank_values

    reciprocal_rank = np.zeros_like(preference, dtype=float)
    reciprocal_rank[valid] = 1.0 / ranks[valid]

    scarcity = reciprocal_rank + reciprocal_rank.T
    scarcity_max = np.max(scarcity)
    if scarcity_max > 0:
        scarcity /= scarcity_max

    score = 0.65 * preference + 0.35 * scarcity
    score = 0.5 * (score + score.T)

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
