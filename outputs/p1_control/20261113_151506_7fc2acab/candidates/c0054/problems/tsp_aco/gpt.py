import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    # Require an edge to be valid in both directions before using it.
    valid &= valid.T
    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    safe_d = np.where(valid, d, np.inf)

    nearest_index = np.argmin(safe_d, axis=1)
    nearest = safe_d[np.arange(n), nearest_index]
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size == 0:
        return np.zeros_like(d, dtype=float)

    positive_distances = d[valid & (d > 0)]
    if positive_distances.size:
        global_scale = float(np.median(positive_distances))
    else:
        global_scale = float(np.median(finite_nearest))

    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    positive_nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    fallback_scale = (
        float(np.median(positive_nearest))
        if positive_nearest.size
        else global_scale
    )
    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        fallback_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    short_edge = np.exp(-safe_d / global_scale)
    endpoint_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_edge = np.exp(-safe_d / endpoint_scale)

    mutual_best = (
        nearest_index[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_index[None, :] == np.arange(n)[:, None]
    )

    score = (
        0.50 * short_edge
        + 0.35 * relative_edge
        + 0.15 * mutual_best.astype(float)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
