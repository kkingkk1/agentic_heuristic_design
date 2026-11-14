import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    positive_values = d[finite_positive]
    fallback = float(np.max(positive_values))
    invalid_value = fallback * 10.0

    d[~np.isfinite(d)] = invalid_value
    d[d < 0] = invalid_value
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps

    nearest_index = np.argmin(d, axis=1)
    row_min = np.min(d, axis=1)
    row_min = np.maximum(row_min, eps)

    inverse_distance = 1.0 / np.maximum(d, eps)
    row_inverse_max = np.max(inverse_distance, axis=1, keepdims=True)
    inverse_distance /= np.maximum(row_inverse_max, eps)

    row_shortness = row_min[:, None] / np.maximum(d, eps)
    row_shortness = np.clip(row_shortness, 0.0, 1.0)

    endpoint_shortness = np.sqrt(
        row_shortness * row_shortness.T
    )

    k = min(4, n - 1)
    top_indices = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    top_mask = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    top_mask[rows, top_indices] = 1.0

    neighborhood_support = 0.5 * (top_mask + top_mask.T)

    mutual_nearest = (
        nearest_index[nearest_index] == np.arange(n)
    ).astype(float)
    mutual_matrix = np.zeros((n, n), dtype=float)
    mutual_matrix[
        np.arange(n),
        nearest_index,
    ] = mutual_nearest

    heuristic = (
        0.45 * inverse_distance
        + 0.30 * endpoint_shortness
        + 0.15 * neighborhood_support
        + 0.10 * mutual_matrix
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
