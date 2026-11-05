import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be nonnegative")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    closeness = 1.0 / (d + scale * 1e-12)

    row_best = np.min(d, axis=1)
    row_second = np.partition(d, 1, axis=1)[:, 1]
    regret = np.maximum(row_second[:, None] - d, 0.0)
    regret += np.maximum(row_second[None, :] - d, 0.0)
    regret /= (row_second[:, None] + row_second[None, :] + scale * 1e-12)

    neighborhood = np.exp(-d / scale)
    np.fill_diagonal(neighborhood, 0.0)
    neighborhood_sum = neighborhood.sum(axis=1, keepdims=True)
    neighborhood_sum = np.maximum(neighborhood_sum, np.finfo(float).eps)
    transition = neighborhood / neighborhood_sum

    bridge = transition @ transition.T
    bridge = bridge / (np.max(bridge) + np.finfo(float).eps)

    def normalize(x: np.ndarray) -> np.ndarray:
        y = np.array(x, dtype=float, copy=True)
        finite = np.isfinite(y)
        if not np.any(finite):
            return np.zeros_like(y)
        lo = np.min(y[finite])
        hi = np.max(y[finite])
        if hi - lo <= np.finfo(float).eps:
            return np.ones_like(y)
        return (y - lo) / (hi - lo)

    score = (
        0.50 * normalize(closeness)
        + 0.30 * normalize(regret)
        + 0.20 * normalize(bridge)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    return np.maximum(score, 0.0)
