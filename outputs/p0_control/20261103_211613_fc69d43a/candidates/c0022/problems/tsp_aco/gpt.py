import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D NumPy array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    distances = distance_matrix.copy()
    diagonal = np.eye(n, dtype=bool)

    finite_off_diagonal = np.isfinite(distances) & ~diagonal
    if not np.any(finite_off_diagonal):
        raise ValueError("distance_matrix must contain finite off-diagonal distances")

    finite_values = distances[finite_off_diagonal]
    fallback = float(np.max(finite_values))

    distances[~np.isfinite(distances)] = fallback
    distances = np.maximum(distances, 0.0)

    positive_rows = distances.copy()
    positive_rows[diagonal] = np.nan
    positive_rows[positive_rows <= 0.0] = np.nan

    row_scales = np.nanmedian(positive_rows, axis=1)
    global_scale = float(np.median(finite_values[finite_values > 0.0]))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    row_scales = np.where(
        np.isfinite(row_scales) & (row_scales > 0.0),
        row_scales,
        global_scale,
    )

    pair_scale = np.sqrt(row_scales[:, None] * row_scales[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    relative_distance = distances / pair_scale

    local_closeness = np.exp(-relative_distance)

    endpoint_balance = np.minimum(
        np.exp(-distances / row_scales[:, None]),
        np.exp(-distances / row_scales[None, :]),
    )

    scores = local_closeness * (0.75 + 0.25 * endpoint_balance)
    scores[diagonal] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores
