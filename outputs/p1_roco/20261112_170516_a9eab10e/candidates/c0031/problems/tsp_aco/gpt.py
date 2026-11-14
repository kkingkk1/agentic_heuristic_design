import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric graph-diffusion prior for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0)
    finite &= ~np.eye(n, dtype=bool)

    if not np.any(finite):
        return result

    finite_values = distances[finite]
    fallback = float(np.median(finite_values))
    if not np.isfinite(fallback) or fallback < 0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = 0.5 * (d + d.T)
    valid = finite & finite.T
    np.fill_diagonal(d, 0.0)

    positive = d[valid & (d > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    bandwidth = np.empty(n, dtype=float)
    for i in range(n):
        values = d[i, valid[i]]
        positive_values = values[values > 0]
        if positive_values.size:
            bandwidth[i] = np.median(positive_values)
        elif values.size:
            bandwidth[i] = max(float(np.mean(values)), global_scale)
        else:
            bandwidth[i] = global_scale

    bandwidth = np.maximum(bandwidth, np.finfo(float).eps)

    pair_bandwidth = np.sqrt(
        bandwidth[:, None] * bandwidth[None, :]
    )

    affinity = np.exp(
        -np.square(d / np.maximum(pair_bandwidth, np.finfo(float).eps))
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    degree = affinity.sum(axis=1)
    degree = np.maximum(degree, np.finfo(float).eps)

    normalized = affinity / np.sqrt(degree[:, None] * degree[None, :])
    closure = normalized @ normalized
    closure[~np.isfinite(closure)] = 0.0
    np.fill_diagonal(closure, 0.0)

    closure_max = float(np.max(closure)) if closure.size else 0.0
    if closure_max > 0:
        closure /= closure_max

    direct = affinity.copy()
    direct_max = float(np.max(direct)) if direct.size else 0.0
    if direct_max > 0:
        direct /= direct_max

    global_term = np.exp(-d / global_scale)
    global_term[~valid] = 0.0
    np.fill_diagonal(global_term, 0.0)

    score = (
        0.45 * direct
        + 0.40 * closure
        + 0.15 * global_term
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0:
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
