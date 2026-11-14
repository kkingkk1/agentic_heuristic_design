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

    if not np.any(valid):
        return heuristic

    inverse_distance = np.zeros_like(distance_matrix, dtype=float)
    np.divide(
        1.0,
        distance_matrix,
        out=inverse_distance,
        where=valid,
    )

    row_scale = np.max(inverse_distance, axis=1, keepdims=True)
    local_preference = np.divide(
        inverse_distance,
        row_scale,
        out=np.zeros_like(inverse_distance),
        where=row_scale > 0,
    )

    eps = np.finfo(float).eps
    endpoint_support = np.sqrt(
        local_preference * local_preference.T
    )
    endpoint_support = np.nan_to_num(
        endpoint_support,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    heuristic = (
        0.70 * local_preference
        + 0.15 * local_preference.T
        + 0.15 * endpoint_support
    )

    heuristic[~valid] = 0.0
    np.fill_diagonal(heuristic, 0.0)

    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return np.maximum(heuristic, eps * valid.astype(float))
