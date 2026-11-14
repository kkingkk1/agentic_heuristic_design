import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    heuristic = np.zeros((n, n), dtype=float)

    if n <= 1:
        return heuristic

    valid = np.isfinite(distance_matrix) & (distance_matrix > 0)
    valid &= ~np.eye(n, dtype=bool)

    inverse_distance = np.zeros_like(distance_matrix)
    np.divide(1.0, distance_matrix, out=inverse_distance, where=valid)

    row_scale = inverse_distance.max(axis=1, keepdims=True)
    np.divide(
        inverse_distance,
        row_scale,
        out=heuristic,
        where=row_scale > 0,
    )

    heuristic[~valid] = 0.0
    return heuristic
