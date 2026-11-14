import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    distance = 0.5 * (d + d.T)
    np.fill_diagonal(distance, np.inf)

    positive = distance[np.isfinite(distance) & (distance > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_distance = np.maximum(distance, np.finfo(float).eps)

    inverse_score = 1.0 / safe_distance
    inverse_score[~np.isfinite(inverse_score)] = 0.0

    exponential_score = np.exp(-np.minimum(distance / scale, 50.0))
    exponential_score[~np.isfinite(exponential_score)] = 0.0

    order = np.argsort(distance, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0

    nearest = order[:, 0]
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    score = (
        0.45 * inverse_score / (np.max(inverse_score, axis=1, keepdims=True) + 1e-12)
        + 0.30 * exponential_score
        + 0.25 * rank_score
        + 0.50 * mutual_nearest
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
