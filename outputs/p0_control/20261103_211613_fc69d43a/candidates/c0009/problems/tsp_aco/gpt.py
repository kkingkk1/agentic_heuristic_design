import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square NumPy array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    valid_off_diagonal = finite.copy()
    np.fill_diagonal(valid_off_diagonal, False)

    positive = distances[finite & (distances > 0)]
    if positive.size == 0:
        return scores

    scale = max(float(np.median(positive)), np.finfo(float).eps)
    safe_distances = np.where(finite, np.maximum(distances, 0.0), np.inf)

    row_values = np.where(valid_off_diagonal, safe_distances, np.nan)
    col_values = row_values.T

    row_median = np.nanmedian(row_values, axis=1)
    col_median = np.nanmedian(col_values, axis=1)

    row_median = np.where(np.isfinite(row_median), row_median, scale)
    col_median = np.where(np.isfinite(col_median), col_median, scale)

    row_contrast = np.divide(
        row_median[:, None] - safe_distances,
        row_median[:, None] + scale,
        out=np.zeros_like(distances),
        where=np.isfinite(safe_distances),
    )
    col_contrast = np.divide(
        col_median[None, :] - safe_distances,
        col_median[None, :] + scale,
        out=np.zeros_like(distances),
        where=np.isfinite(safe_distances),
    )

    local_prominence = np.exp(
        np.clip(2.0 * row_contrast + 2.0 * col_contrast, -20.0, 20.0)
    )
    inverse_distance = 1.0 / (safe_distances + scale)
    scores = inverse_distance * local_prominence

    scores[~valid_off_diagonal] = 0.0
    scores[~np.isfinite(scores)] = 0.0

    maximum = np.max(scores)
    if maximum > 0.0:
        scores /= maximum

    return scores
