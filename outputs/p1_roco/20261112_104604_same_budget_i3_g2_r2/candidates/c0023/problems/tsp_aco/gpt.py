import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    positive = distances[valid]
    if positive.size == 0:
        return scores

    global_scale = np.median(positive)
    global_scale = max(float(global_scale), np.finfo(float).eps)

    safe_distances = np.where(valid, distances, np.inf)

    row_scales = np.median(
        np.where(valid, distances, np.nan),
        axis=1
    )
    row_scales = np.where(
        np.isfinite(row_scales) & (row_scales > 0),
        row_scales,
        global_scale,
    )

    local_affinity = np.zeros_like(distances, dtype=float)
    local_affinity[valid] = np.exp(
        -distances[valid]
        / np.sqrt(row_scales[:, None][valid] * row_scales[None, :][valid])
    )

    symmetric_local = np.sqrt(
        np.maximum(local_affinity, 0.0)
        * np.maximum(local_affinity.T, 0.0)
    )
    symmetric_local[~valid] = 0.0

    inverse_distance = np.zeros_like(distances, dtype=float)
    inverse_distance[valid] = 1.0 / (
        distances[valid] + np.finfo(float).eps
    )
    inverse_scale = np.median(inverse_distance[valid])
    if inverse_scale > 0:
        inverse_distance /= inverse_scale
    inverse_distance[~valid] = 0.0

    first = np.min(safe_distances, axis=1)
    first_index = np.argmin(safe_distances, axis=1)

    masked_without_first = safe_distances.copy()
    masked_without_first[
        np.arange(n),
        first_index
    ] = np.inf
    second = np.min(masked_without_first, axis=1)

    second = np.where(
        np.isfinite(second),
        second,
        first
    )

    alternative_i = np.where(
        first_index[:, None] == np.arange(n)[None, :],
        second[:, None],
        first[:, None],
    )
    alternative_j = np.where(
        first_index[None, :] == np.arange(n)[:, None],
        second[None, :],
        first[None, :],
    )

    savings = alternative_i + alternative_j - distances
    savings = np.maximum(savings, 0.0)
    savings[~valid] = 0.0

    positive_savings = savings[valid]
    if positive_savings.size > 0 and np.max(positive_savings) > 0:
        savings /= np.max(positive_savings)

    scores = (
        0.45 * inverse_distance
        + 0.30 * symmetric_local
        + 0.25 * savings
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores
