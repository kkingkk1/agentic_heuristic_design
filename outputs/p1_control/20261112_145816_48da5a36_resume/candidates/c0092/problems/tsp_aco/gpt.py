import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    distances = 0.5 * (distances + distances.T)

    valid = (
        np.isfinite(distances)
        & (distances > 0.0)
    )
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive_values = distances[valid]
    temperature = float(np.median(positive_values))
    temperature = max(temperature, np.finfo(float).eps)

    minimum = float(np.min(positive_values))
    shifted = np.maximum(distances - minimum, 0.0)

    kernel = np.zeros((n, n), dtype=float)
    kernel[valid] = np.exp(
        -np.minimum(shifted[valid] / temperature, 700.0)
    )

    active = np.any(valid, axis=1)
    active_indices = np.flatnonzero(active)
    active_kernel = kernel[np.ix_(active_indices, active_indices)]

    if active_kernel.size == 0:
        return result

    m = active_kernel.shape[0]
    left_scale = np.ones(m, dtype=float)
    right_scale = np.ones(m, dtype=float)
    eps = np.finfo(float).eps

    for _ in range(160):
        row_mass = active_kernel @ right_scale
        left_scale = np.divide(
            1.0,
            np.maximum(row_mass, eps),
        )

        column_mass = active_kernel.T @ left_scale
        right_scale = np.divide(
            1.0,
            np.maximum(column_mass, eps),
        )

    balanced = (
        left_scale[:, None]
        * active_kernel
        * right_scale[None, :]
    )

    balanced = 0.5 * (balanced + balanced.T)
    balanced[~np.isfinite(balanced)] = 0.0

    result[np.ix_(active_indices, active_indices)] = balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    return result.astype(float, copy=False)
