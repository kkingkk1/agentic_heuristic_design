import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a two-dimensional array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    distances = np.array(distance_matrix, dtype=float, copy=True)
    distances[~np.isfinite(distances)] = np.inf
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[np.isfinite(distances)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_distances = np.maximum(distances, 0.0)

    inverse_term = 1.0 / (1.0 + safe_distances / scale)
    inverse_term[~np.isfinite(inverse_term)] = 0.0

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        order = np.argsort(safe_distances[i], kind="stable")
        rank = 0
        for j in order:
            if i == j or not np.isfinite(safe_distances[i, j]):
                continue
            ranks[i, j] = rank
            rank += 1

    rank_term = np.exp(-ranks / max(1.0, np.sqrt(n)))
    rank_term[~np.isfinite(rank_term)] = 0.0
    rank_term[np.diag_indices(n)] = 0.0

    nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid = np.flatnonzero(np.isfinite(safe_distances[i]))
        valid = valid[valid != i]
        if valid.size:
            nearest[i, valid[np.argmin(safe_distances[i, valid])]] = 1.0

    mutual_nearest = nearest * nearest.T

    heuristic = (
        0.55 * inverse_term
        + 0.30 * rank_term
        + 0.15 * mutual_nearest
    )

    heuristic = 0.5 * (heuristic + heuristic.T)
    heuristic[~np.isfinite(heuristic)] = 0.0
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
