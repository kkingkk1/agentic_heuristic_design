import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a two-dimensional array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    if np.any(np.isfinite(distance_matrix) & (distance_matrix < 0)):
        raise ValueError("distance_matrix must contain non-negative distances")

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & ~diagonal

    if not np.any(valid):
        return scores

    finite_values = distance_matrix[valid]
    global_scale = float(np.median(finite_values))
    if not np.isfinite(global_scale) or global_scale <= 0:
        positive_values = finite_values[finite_values > 0]
        global_scale = (
            float(np.median(positive_values))
            if positive_values.size > 0
            else 1.0
        )
    global_scale = max(global_scale, np.finfo(float).eps)

    replacement = max(float(np.max(finite_values)), global_scale) * 10.0 + global_scale
    distances = np.where(valid, distance_matrix, replacement)

    row_scales = np.empty(n, dtype=float)
    for i in range(n):
        row_values = distance_matrix[i, valid[i]]
        if row_values.size == 0:
            row_scales[i] = global_scale
        else:
            scale = float(np.median(row_values))
            row_scales[i] = scale if np.isfinite(scale) and scale > 0 else global_scale

    row_scales = np.maximum(row_scales, np.finfo(float).eps)

    local_attraction = np.exp(
        -distances / row_scales[:, None]
    )
    local_attraction[~valid] = 0.0

    row_mass = local_attraction.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    normalized_attraction = local_attraction / row_mass

    pair_attraction = 0.5 * (
        local_attraction / row_scales[:, None]
        + local_attraction.T / row_scales[None, :]
    )
    pair_attraction[~valid] = 0.0
    pair_max = np.max(pair_attraction)
    if pair_max > 0:
        pair_attraction /= pair_max

    nearest = np.argmin(distances, axis=1)
    nearest_component = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if valid[i, j]:
            nearest_component[i, j] = 1.0
            nearest_component[j, i] = 1.0

    corridor = np.zeros((n, n), dtype=float)
    if n >= 3:
        best_two_hop = np.full((n, n), replacement, dtype=float)
        for k in range(n):
            candidate = distances[:, k, None] + distances[k, None, :]
            allowed = np.ones((n, n), dtype=bool)
            allowed[k, :] = False
            allowed[:, k] = False
            allowed[diagonal] = False
            best_two_hop = np.minimum(
                best_two_hop,
                np.where(allowed, candidate, replacement),
            )

        pair_scale = 0.5 * (
            row_scales[:, None] + row_scales[None, :]
        )
        corridor = np.exp(-best_two_hop / np.maximum(pair_scale, np.finfo(float).eps))
        corridor[~valid] = 0.0
        corridor_max = np.max(corridor)
        if corridor_max > 0:
            corridor /= corridor_max

    scores = (
        0.45 * pair_attraction
        + 0.25 * normalized_attraction
        + 0.20 * corridor
        + 0.10 * nearest_component
    )

    scores = 0.5 * (scores + scores.T)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    return scores
