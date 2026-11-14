import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    safe_distances = np.where(valid, distances, np.nan)

    row_scale = np.nanmedian(safe_distances, axis=1)
    finite_row_scales = row_scale[np.isfinite(row_scale) & (row_scale > 0)]
    fallback = (
        float(np.median(finite_row_scales))
        if finite_row_scales.size
        else 1.0
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback,
    )

    inverse_distances = 1.0 / np.where(valid, distances, np.inf)

    endpoint_score = np.sqrt(
        (row_scale[:, None] * inverse_distances)
        * (row_scale[None, :] * inverse_distances)
    )

    positive_distances = distances[valid]
    global_scale = (
        float(np.median(positive_distances))
        if positive_distances.size
        else fallback
    )
    global_score = global_scale * inverse_distances

    score = 0.85 * endpoint_score + 0.15 * global_score
    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0:
        score /= maximum

    return score
