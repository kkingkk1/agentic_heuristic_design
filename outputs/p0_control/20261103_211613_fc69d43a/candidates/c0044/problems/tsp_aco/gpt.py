import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a stable symmetric edge-score matrix for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    if np.any(np.isfinite(distances) & (distances < 0.0)):
        raise ValueError("distance_matrix must contain nonnegative distances")

    valid = np.isfinite(distances)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return scores

    scale = float(np.median(positive))
    epsilon = max(scale * 1e-12, np.finfo(float).eps)

    working = np.where(valid, np.maximum(distances, epsilon), np.inf)
    neighbor_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n - 1)))))

    sorted_rows = np.sort(working, axis=1)
    local_scales = np.mean(sorted_rows[:, :neighbor_count], axis=1)
    local_scales = np.where(
        np.isfinite(local_scales),
        np.maximum(local_scales, epsilon),
        max(scale, epsilon),
    )

    endpoint_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )
    relative_length = distances / np.maximum(endpoint_scale, epsilon)
    relative_length = np.maximum(relative_length, 0.0)

    local_affinity = np.exp(-relative_length)

    row_preference = local_affinity.copy()
    np.fill_diagonal(row_preference, 0.0)

    row_max = np.max(row_preference, axis=1, keepdims=True)
    row_max = np.maximum(row_max, epsilon)
    normalized_preference = row_preference / row_max

    mutual_preference = np.sqrt(
        normalized_preference * normalized_preference.T
    )

    neighborhood_overlap = normalized_preference @ normalized_preference.T
    overlap_scale = np.maximum(
        np.max(neighborhood_overlap, axis=1, keepdims=True),
        epsilon,
    )
    compatibility = neighborhood_overlap / np.sqrt(
        overlap_scale * overlap_scale.T
    )
    compatibility = np.clip(compatibility, 0.0, 1.0)

    reciprocal = 1.0 / np.maximum(distances, epsilon)
    reciprocal /= max(1.0 / max(scale, epsilon), epsilon)

    raw_scores = (
        reciprocal
        * local_affinity
        * (0.5 + 0.5 * mutual_preference)
        * (0.75 + 0.25 * compatibility)
    )

    raw_scores = 0.5 * (raw_scores + raw_scores.T)
    raw_scores[~valid] = 0.0
    np.fill_diagonal(raw_scores, 0.0)
    raw_scores[~np.isfinite(raw_scores)] = 0.0

    maximum = float(np.max(raw_scores))
    if maximum > 0.0:
        scores = raw_scores / maximum

    np.fill_diagonal(scores, 0.0)
    return scores
