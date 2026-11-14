import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0)
    valid &= ~np.eye(n, dtype=bool)

    safe_distances = np.where(valid, distances, np.inf)
    positive = valid & (safe_distances > 0)

    positive_values = safe_distances[positive]
    if positive_values.size:
        scale = np.median(positive_values)
    else:
        scale = 1.0

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[positive] = 1.0 / (safe_distances[positive] + 1e-12)
    inverse_distance = np.minimum(inverse_distance * scale, 1.0)

    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_valid = valid[i]
        candidates = np.flatnonzero(row_valid)
        if candidates.size:
            order = candidates[np.argsort(safe_distances[i, candidates])]
            rank_score[i, order] = 1.0 - (
                np.arange(order.size, dtype=float) / max(order.size - 1, 1)
            )

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    symmetric_distance = np.minimum(inverse_distance, inverse_distance.T)
    symmetric_distance = np.maximum(
        symmetric_distance,
        0.5 * (inverse_distance + inverse_distance.T),
    )

    scores = (
        0.60 * symmetric_distance
        + 0.25 * mutual_rank
        + 0.15 * np.sqrt(np.maximum(rank_score * rank_score.T, 0.0))
    )

    scores = 0.5 * (scores + scores.T)
    scores[~valid & ~valid.T] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
