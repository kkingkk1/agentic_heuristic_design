import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n < 2:
        return np.zeros_like(distances, dtype=float)

    d = np.array(distances, dtype=float, copy=True)
    finite_positive = np.isfinite(d) & (d > 0)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    fallback = np.max(d[finite_positive])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    d[~np.isfinite(d)] = fallback
    d[d < 0] = fallback
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    global_scale = np.median(positive)
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = np.mean(positive)
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    global_score = np.exp(-d / global_scale)
    global_score[~np.isfinite(global_score)] = 0.0

    local_scale = np.min(d, axis=1)
    local_scale[~np.isfinite(local_scale)] = global_scale
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_score = np.exp(
        -d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    )
    relative_score[~np.isfinite(relative_score)] = 0.0

    nearest_count = max(1, min(3, n - 1))
    nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        order = np.argsort(d[i])
        candidates = order[:nearest_count]
        candidates = candidates[np.isfinite(d[i, candidates])]
        nearest[i, candidates] = 1.0

    mutual_bonus = nearest * nearest.T

    score = (
        0.50 * global_score
        + 0.30 * relative_score
        + 0.20 * mutual_bonus
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
