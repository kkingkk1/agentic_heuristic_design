import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be a square 2D array")
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distance_matrix)
    np.fill_diagonal(valid, False)

    finite_edges = distance_matrix[valid]
    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = np.max(finite_edges)
    distances = np.where(valid, distance_matrix, fallback)

    positive_edges = finite_edges[finite_edges > 0]
    if positive_edges.size:
        epsilon = max(
            np.finfo(float).eps,
            np.median(positive_edges) * 1e-12,
        )
    else:
        epsilon = np.finfo(float).eps

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / (distances[valid] + epsilon)

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).eps)
    row_score = inverse_distance / row_max

    col_max = np.max(inverse_distance, axis=0, keepdims=True)
    col_max = np.maximum(col_max, np.finfo(float).eps)
    col_score = inverse_distance / col_max

    nearest = np.argmin(np.where(valid, distances, np.inf), axis=1)
    mutual = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[nearest][:, None] == np.arange(n)[:, None])
    )

    score = (
        0.65 * row_score
        + 0.25 * col_score
        + 0.10 * mutual.astype(float)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score_max = np.max(score, axis=1, keepdims=True)
    score_max = np.maximum(score_max, np.finfo(float).eps)

    return score / score_max
