import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric distance-aware edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative values")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    eps = np.finfo(float).eps
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    safe_d = np.maximum(d, eps)

    inverse = 1.0 / safe_d
    inverse[~np.isfinite(inverse)] = 0.0
    inverse /= np.max(inverse, axis=1, keepdims=True) + eps

    k = min(max(2, int(np.sqrt(n))), n - 1)
    nearest_values = np.partition(safe_d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, eps)

    local_distance = np.sqrt(np.outer(local_scale, local_scale))
    relative = np.exp(-safe_d / local_distance)
    relative[~np.isfinite(relative)] = 0.0

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0

    reciprocal_rank = np.sqrt(rank_score * rank_score.T)

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    edge_mass = inverse / (np.sum(inverse, axis=1, keepdims=True) + eps)
    degree_balance = np.sqrt(edge_mass * edge_mass.T)

    score = (
        0.38 * inverse
        + 0.27 * relative
        + 0.20 * reciprocal_rank
        + 0.15 * mutual
    )
    score *= 0.65 + 0.35 * degree_balance

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
