import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric local-competition edge scores for a TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite_positive = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(finite_positive, False)

    usable = distances[finite_positive]
    if usable.size == 0:
        return scores

    global_scale = float(np.min(usable))
    if not np.isfinite(global_scale) or global_scale <= 0:
        return scores

    row_scales = np.where(
        finite_positive,
        distances,
        np.inf,
    ).min(axis=1)
    row_scales = np.where(
        np.isfinite(row_scales) & (row_scales > 0),
        row_scales,
        global_scale,
    )

    mutual_valid = finite_positive & finite_positive.T
    if not np.any(mutual_valid):
        return scores

    forward = np.where(mutual_valid, distances, 1.0)
    reverse = np.where(mutual_valid, distances.T, 1.0)
    pair_distances = np.sqrt(forward * reverse)

    endpoint_affinity = (
        row_scales[:, None] / pair_distances
    ) * (
        row_scales[None, :] / pair_distances
    )

    scale_min = np.minimum(row_scales[:, None], row_scales[None, :])
    scale_max = np.maximum(row_scales[:, None], row_scales[None, :])
    compatibility = np.divide(
        2.0 * scale_min,
        scale_min + scale_max,
        out=np.zeros_like(pair_distances),
        where=(scale_max > 0),
    )

    raw_scores = endpoint_affinity * compatibility
    raw_scores = np.where(mutual_valid, raw_scores, 0.0)
    np.fill_diagonal(raw_scores, 0.0)

    symmetric_scores = np.sqrt(
        np.maximum(raw_scores, 0.0)
        * np.maximum(raw_scores.T, 0.0)
    )
    symmetric_scores = np.where(mutual_valid, symmetric_scores, 0.0)
    np.fill_diagonal(symmetric_scores, 0.0)

    maximum = float(np.max(symmetric_scores))
    if not np.isfinite(maximum) or maximum <= 0:
        return scores

    scores = symmetric_scores / maximum
    return np.clip(scores, 0.0, 1.0).astype(float, copy=False)
