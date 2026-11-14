import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-prior indicators for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid_direction = np.isfinite(d) & (d > 0)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = valid_direction & valid_direction.T & off_diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    positive = symmetric_d[valid]
    global_scale = float(np.median(positive))
    eps = max(np.finfo(float).eps, global_scale * 1e-12)

    safe_d = np.where(valid, symmetric_d, np.inf)

    row_values = np.where(valid, symmetric_d, np.nan)
    row_scale = np.nanmedian(row_values, axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        global_scale,
    )
    row_scale = np.maximum(row_scale, eps)

    endpoint_scale = np.sqrt(
        row_scale[:, None] * row_scale[None, :]
    )
    normalized_distance = safe_d / endpoint_scale

    short_edge = np.exp(-np.minimum(normalized_distance, 60.0))
    short_edge[~valid] = 0.0

    inverse_distance = np.where(
        valid,
        1.0 / np.maximum(safe_d, eps),
        0.0,
    )
    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    row_preference = inverse_distance / row_max
    mutual_preference = np.sqrt(
        row_preference * row_preference.T
    )

    nearest = np.argmin(safe_d, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j < n and nearest[j] == i and valid[i, j]:
            mutual_nearest[i, j] = 1.0

    competition = np.sum(short_edge, axis=1)
    competition = np.maximum(competition, eps)
    diversity = 1.0 / np.sqrt(
        competition[:, None] * competition[None, :]
    )
    diversity *= valid

    score = (
        0.50 * short_edge
        + 0.30 * mutual_preference
        + 0.12 * mutual_nearest
        + 0.08 * diversity
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
