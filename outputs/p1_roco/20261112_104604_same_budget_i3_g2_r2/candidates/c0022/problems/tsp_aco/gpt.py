import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive = distances[valid & (distances > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    zero_mask = valid & (distances == 0.0)

    inverse = np.zeros_like(distances, dtype=float)
    inverse[valid] = 1.0 / np.maximum(distances[valid], global_scale * 1e-12)
    if np.any(valid):
        inverse /= np.max(inverse[valid])

    row_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        row_positive = distances[i, valid[i] & (distances[i] > 0.0)]
        if row_positive.size:
            row_scale[i] = max(float(np.median(row_positive)), global_scale * 1e-6)

    local = np.zeros_like(distances, dtype=float)
    local[valid] = np.exp(
        -distances[valid]
        / np.maximum(row_scale[np.nonzero(valid)[0]], np.finfo(float).eps)
    )

    local_symmetric = np.sqrt(
        np.maximum(local, 0.0) * np.maximum(local.T, 0.0)
    )

    global_contrast = np.zeros_like(distances, dtype=float)
    global_contrast[valid] = np.exp(
        -distances[valid] / global_scale
    )

    nearest_two = np.zeros_like(distances, dtype=bool)
    k = min(2, max(n - 1, 1))
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(distances[i, candidates], kind="stable")]
            nearest_two[i, order[:k]] = True

    mutual_two_nearest = nearest_two & nearest_two.T
    two_neighbor_bonus = mutual_two_nearest.astype(float)
    two_neighbor_bonus[zero_mask] = 1.0

    scores = (
        0.55 * inverse
        + 0.25 * local_symmetric
        + 0.15 * global_contrast
        + 0.05 * two_neighbor_bonus
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0:
        scores /= maximum

    return scores
