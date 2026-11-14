import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    valid_input = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid_input, False)

    valid = valid_input & valid_input.T
    if not np.any(valid):
        return scores

    symmetric_distances = 0.5 * (distances + distances.T)
    symmetric_distances[~valid] = np.inf

    positive_distances = symmetric_distances[valid]
    global_scale = np.median(positive_distances)

    if not np.isfinite(global_scale) or global_scale <= 0:
        return scores

    inverse_distance = np.zeros_like(symmetric_distances, dtype=float)
    inverse_distance[valid] = (
        global_scale / (symmetric_distances[valid] + global_scale)
    )

    local_scales = np.full(n, global_scale, dtype=float)
    for i in range(n):
        row_values = symmetric_distances[i, valid[i]]
        if row_values.size:
            local_scale = np.median(row_values)
            if np.isfinite(local_scale) and local_scale > 0:
                local_scales[i] = local_scale

    neighborhood = np.zeros_like(symmetric_distances, dtype=float)
    for i in range(n):
        row_valid = valid[i]
        neighborhood[i, row_valid] = np.exp(
            -symmetric_distances[i, row_valid] / local_scales[i]
        )

    neighborhood_similarity = np.sqrt(
        neighborhood * neighborhood.T
    )

    density_ratio = np.minimum(
        local_scales[:, None] / (local_scales[None, :] + 1e-12),
        local_scales[None, :] / (local_scales[:, None] + 1e-12),
    )
    density_similarity = np.sqrt(np.clip(density_ratio, 0.0, 1.0))

    scores = (
        0.70 * inverse_distance
        + 0.20 * neighborhood_similarity
        + 0.10 * density_similarity
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
