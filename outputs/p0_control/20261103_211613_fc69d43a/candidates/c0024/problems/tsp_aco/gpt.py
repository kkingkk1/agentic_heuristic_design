import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    d = distance_matrix.copy()
    np.fill_diagonal(d, np.nan)

    finite = np.isfinite(d)
    if not np.any(finite):
        return scores

    large = np.nanmax(np.where(finite, d, np.nan))
    if not np.isfinite(large):
        return scores

    d = np.where(finite, d, large * 10.0 + 1.0)
    d = np.maximum(d, 0.0)

    row_scale = np.partition(d, min(2, n - 1), axis=1)[:, min(2, n - 1)]
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    direct = np.exp(-d / row_scale[:, None])

    reverse_scale = np.maximum(row_scale, np.finfo(float).eps)
    mutual = np.exp(-d / reverse_scale[None, :])

    affinity = direct * mutual

    neighborhood = np.exp(-d / row_scale[:, None])
    np.fill_diagonal(neighborhood, 0.0)

    overlap = neighborhood @ neighborhood.T
    overlap_scale = np.maximum(
        np.max(overlap, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    compatibility = overlap / overlap_scale

    scores = affinity * (0.5 + 0.5 * compatibility)

    np.fill_diagonal(scores, 0.0)
    scores[~finite] = 0.0

    maximum = np.max(scores)
    if maximum > 0.0:
        scores /= maximum

    return scores
