import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    positive_values = d[finite_positive]

    if positive_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(positive_values) * 10.0)
    work = np.where(finite_positive, d, fallback)
    np.fill_diagonal(work, np.inf)

    row_sorted = np.sort(work, axis=1)
    nearest = row_sorted[:, 0]
    scale = np.maximum(nearest, np.finfo(float).eps)

    relative_distance = work / scale[:, None]
    inverse_score = 1.0 / np.maximum(relative_distance, np.finfo(float).eps)
    inverse_score = np.clip(inverse_score, 0.0, 1.0)

    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))

    nearest_indices = np.argmin(work, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_indices[i]
        if j != i and nearest_indices[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    edge_score = (
        0.55 * np.sqrt(inverse_score * inverse_score.T)
        + 0.35 * np.sqrt(rank_score * rank_score.T)
        + 0.10 * mutual
    )

    valid = finite_positive & ~np.eye(n, dtype=bool)
    edge_score *= valid
    edge_score = 0.5 * (edge_score + edge_score.T)

    row_max = np.max(edge_score, axis=1, keepdims=True)
    edge_score = edge_score / np.maximum(row_max, np.finfo(float).eps)

    np.fill_diagonal(edge_score, 0.0)
    return edge_score.astype(float, copy=False)
