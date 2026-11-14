import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    if not np.any(finite_positive):
        return np.zeros_like(distances, dtype=float)

    fallback = float(np.max(distances[finite_positive]))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.array(distances, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = fallback
    clean[clean <= 0] = fallback
    np.fill_diagonal(clean, np.inf)

    valid = np.isfinite(clean)

    inverse = np.zeros_like(clean, dtype=float)
    inverse[valid] = 1.0 / np.maximum(clean[valid], np.finfo(float).eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    inverse /= row_max

    order = np.argsort(clean, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = np.zeros_like(clean, dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])

    local_scale = np.min(np.where(valid, clean, np.inf), axis=1)
    local_scale[~np.isfinite(local_scale)] = fallback
    closeness = np.zeros_like(clean, dtype=float)
    closeness[valid] = np.exp(
        -(clean[valid] - local_scale[np.nonzero(valid)[0]])
        / (local_scale[np.nonzero(valid)[0]] + np.finfo(float).eps)
    )

    row_max = np.max(closeness, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    closeness /= row_max

    nearest = np.argmin(clean, axis=1)
    mutual = np.zeros_like(clean, dtype=float)
    for i, j in enumerate(nearest):
        if j != i and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    local_score = (
        0.55 * inverse
        + 0.25 * closeness
        + 0.10 * rank_score
        + 0.10 * mutual
    )
    local_score[~valid] = 0.0
    np.fill_diagonal(local_score, 0.0)

    score = 0.5 * (local_score + local_score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
