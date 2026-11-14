import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite_positive = np.isfinite(distance_matrix) & (distance_matrix > 0)
    np.fill_diagonal(finite_positive, False)

    symmetric_valid = finite_positive & finite_positive.T
    if not np.any(symmetric_valid):
        return scores

    symmetric_distances = 0.5 * (
        np.where(finite_positive, distances, 0.0)
        + np.where(finite_positive.T, distances.T, 0.0)
    )

    symmetric_distances = np.where(
        symmetric_valid,
        symmetric_distances,
        np.inf,
    )
    np.fill_diagonal(symmetric_distances, np.inf)

    valid_values = symmetric_distances[symmetric_valid]
    global_scale = np.median(valid_values)
    global_scale = max(global_scale, np.finfo(float).eps)

    inverse_distance = np.zeros_like(scores)
    inverse_distance[symmetric_valid] = (
        global_scale / (symmetric_distances[symmetric_valid] + global_scale)
    )

    row_scales = np.full(n, global_scale, dtype=float)
    for i in range(n):
        row_values = symmetric_distances[i, symmetric_valid[i]]
        if row_values.size:
            row_scales[i] = max(
                np.median(row_values),
                np.finfo(float).eps,
            )

    local_affinity = np.zeros_like(scores)
    local_affinity[symmetric_valid] = np.exp(
        -symmetric_distances[symmetric_valid]
        / row_scales[np.newaxis, :].T.repeat(n, axis=1)[symmetric_valid]
    )

    row_maxima = np.max(local_affinity, axis=1, keepdims=True)
    row_maxima = np.maximum(row_maxima, np.finfo(float).eps)
    local_affinity /= row_maxima

    neighborhood_support = np.sqrt(
        np.maximum(local_affinity, 0.0)
        * np.maximum(local_affinity.T, 0.0)
    )

    scale_support = np.zeros_like(scores)
    scale_support[symmetric_valid] = np.exp(
        -symmetric_distances[symmetric_valid] / global_scale
    )

    scores = (
        0.70 * inverse_distance
        + 0.20 * neighborhood_support
        + 0.10 * scale_support
    )

    scores[~symmetric_valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0:
        scores /= maximum

    return scores
