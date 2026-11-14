import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    distances = np.maximum(distance_matrix, 0.0).copy()
    np.fill_diagonal(distances, np.inf)

    finite_distances = distances[np.isfinite(distances)]
    scale = np.median(finite_distances) if finite_distances.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse_distance = 1.0 / (distances + scale * 1e-6)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_max = inverse_distance.max(axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0
    inverse_score = inverse_distance / row_max

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values

    rank_score = 1.0 - ranks / max(n - 1, 1)
    rank_score[~np.isfinite(distances)] = 0.0

    nearest = order[:, 0]
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j < n and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    score = 0.55 * inverse_score + 0.35 * rank_score + 0.10 * mutual_nearest
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
