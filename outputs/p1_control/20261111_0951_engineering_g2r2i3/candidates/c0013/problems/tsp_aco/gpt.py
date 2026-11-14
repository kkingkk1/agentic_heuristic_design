import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    if n <= 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()
    finite_positive = np.isfinite(d) & (d > 0)

    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(d[finite_positive]))
    invalid_value = fallback * 10.0

    d[~finite_positive] = invalid_value
    np.fill_diagonal(d, invalid_value)

    inverse_distance = 1.0 / d
    row_scale = np.max(inverse_distance, axis=1, keepdims=True)

    local_preference = np.divide(
        inverse_distance,
        row_scale,
        out=np.zeros_like(inverse_distance),
        where=row_scale > 0,
    )

    endpoint_support = np.sqrt(
        local_preference * local_preference.T
    )

    heuristic = (
        0.75 * local_preference
        + 0.25 * endpoint_support
    )

    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
