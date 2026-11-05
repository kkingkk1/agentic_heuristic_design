import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return contrastive local-scale edge scores for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D NumPy array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0)
    off_diagonal = ~np.eye(n, dtype=bool)
    finite_values = distances[valid & off_diagonal]

    if finite_values.size == 0:
        return np.zeros_like(distances, dtype=float)

    fallback = float(np.max(finite_values))
    safe_distances = np.where(valid, distances, fallback)
    np.fill_diagonal(safe_distances, fallback)

    positive_values = safe_distances[off_diagonal]
    scale_floor = max(float(np.median(positive_values)) * 1e-12, np.finfo(float).eps)

    local_scale = np.median(
        np.where(off_diagonal, safe_distances, np.nan),
        axis=1
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > scale_floor),
        local_scale,
        max(float(np.median(positive_values)), scale_floor)
    )

    endpoint_scale = 0.5 * (
        local_scale[:, None] + local_scale[None, :]
    )
    contrast = safe_distances / np.maximum(endpoint_scale, scale_floor)

    global_scale = max(float(np.median(positive_values)), scale_floor)
    reciprocal_term = 1.0 / np.maximum(safe_distances, scale_floor)
    contrast_term = np.exp(-contrast)
    scale_term = 1.0 / np.maximum(
        0.5 * (safe_distances + endpoint_scale),
        scale_floor
    )

    scores = (
        0.45 * reciprocal_term / global_scale
        + 0.35 * contrast_term
        + 0.20 * scale_term * global_scale
    )

    scores = 0.5 * (scores + scores.T)
    scores[~off_diagonal] = 0.0
    scores[~np.isfinite(scores)] = 0.0

    return scores
