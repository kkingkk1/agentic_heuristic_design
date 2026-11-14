import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    distances = np.array(distance_matrix, copy=True)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[np.isfinite(distances)]
    if finite_values.size == 0:
        return scores

    fallback = max(float(np.max(finite_values)), 1.0) * 2.0
    distances[~np.isfinite(distances)] = fallback
    distances = np.maximum(distances, 0.0)

    # Symmetrize because an undirected TSP edge has one shared priority.
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse_score = 1.0 / (1.0 + distances / scale)

    order = np.argsort(distances, axis=1)
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]
    rank_score = np.exp(-ranks / max(1.0, n / 4.0))
    np.fill_diagonal(rank_score, 0.0)

    nearest = order[:, 0]
    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if nearest[j] == i:
            mutual_bonus[i, j] = 1.0
            mutual_bonus[j, i] = 1.0

    scores = 0.55 * inverse_score + 0.35 * rank_score + 0.10 * mutual_bonus
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
