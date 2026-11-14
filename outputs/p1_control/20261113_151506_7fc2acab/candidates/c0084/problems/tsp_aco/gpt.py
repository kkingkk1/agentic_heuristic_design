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

    safe_d = np.where(valid, d, np.inf)
    nearest = np.min(safe_d, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size == 0:
        return np.zeros_like(d, dtype=float)

    positive = d[valid & (d > 0)]
    if positive.size:
        global_scale = float(np.median(finite_nearest))
        if not np.isfinite(global_scale) or global_scale <= 0:
            global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        global_scale,
    )

    short_edge = np.exp(-safe_d / global_scale)
    relative_edge = np.exp(
        -safe_d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    )

    k = min(n - 1, max(2, int(np.sqrt(n))))
    candidate_distances = np.partition(safe_d, k - 1, axis=1)[:, k - 1]
    candidate_mask = valid & (safe_d <= candidate_distances[:, None])

    mutual_mask = candidate_mask & candidate_mask.T
    diverse_support = candidate_mask.astype(float) + candidate_mask.T.astype(float)
    diverse_support = np.minimum(diverse_support, 1.0)

    score = (
        0.45 * short_edge
        + 0.30 * relative_edge
        + 0.15 * diverse_support
        + 0.10 * mutual_mask.astype(float)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
