import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a bidirectional continuation edge-score matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances) & ~np.eye(n, dtype=bool)
    if np.any(distances[valid] < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    safe = np.where(valid, distances, np.inf)

    row_scale = np.min(safe, axis=1)
    col_scale = np.min(safe, axis=0)

    finite_row_scales = row_scale[np.isfinite(row_scale)]
    finite_col_scales = col_scale[np.isfinite(col_scale)]

    fallback_row = (
        float(np.median(finite_row_scales))
        if finite_row_scales.size
        else 1.0
    )
    fallback_col = (
        float(np.median(finite_col_scales))
        if finite_col_scales.size
        else 1.0
    )

    row_scale = np.where(np.isfinite(row_scale), row_scale, fallback_row)
    col_scale = np.where(np.isfinite(col_scale), col_scale, fallback_col)

    eps = np.finfo(float).eps
    direct = 1.0 / np.maximum(safe, eps)

    forward_weight = 0.65
    backward_weight = 0.65
    baseline = 0.35

    continuation_from_j = 1.0 / np.maximum(
        row_scale[:, None] + 0.75 * safe,
        eps,
    )
    continuation_to_i = 1.0 / np.maximum(
        0.75 * safe + col_scale[None, :],
        eps,
    )

    row_best = np.max(continuation_from_j, axis=1)
    col_best = np.max(continuation_to_i, axis=0)

    forward_fit = continuation_from_j / np.maximum(row_best[:, None], eps)
    backward_fit = continuation_to_i / np.maximum(col_best[None, :], eps)

    forward_factor = baseline + forward_weight * forward_fit
    backward_factor = baseline + backward_weight * backward_fit

    scores = direct * forward_factor * backward_factor
    scores[~valid] = 0.0

    finite_scores = np.isfinite(scores)
    scores[~finite_scores] = 0.0

    return scores
