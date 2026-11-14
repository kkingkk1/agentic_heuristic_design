import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric stochastic-sampling priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    positive = d[valid & (d > 0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    safe_d = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps

    # Local neighborhood scale from the shortest incident edges.
    k = max(1, min(n - 1, int(np.sqrt(n))))
    sorted_d = np.sort(safe_d, axis=1)
    local_scale = np.median(sorted_d[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    finite_d = np.where(valid, d, global_scale * 100.0)

    # Direct distance preference.
    inverse = 1.0 / (finite_d + global_scale * 1e-6)

    # Reciprocal local rank preference.
    order = np.argsort(finite_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    for i in range(n):
        ranks[i, order[i]] = np.arange(n, dtype=float)
    rank_signal = 1.0 / (1.0 + ranks)

    # Relative attractiveness at both endpoints.
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = np.exp(
        -np.minimum(finite_d / (endpoint_scale + eps), 50.0)
    )

    # Mild preference for globally short alternatives.
    global_distance = np.exp(
        -np.minimum(finite_d / (global_scale + eps), 50.0)
    )

    # Mutual-nearest-neighbor reinforcement.
    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    def row_normalize(x: np.ndarray) -> np.ndarray:
        x = np.where(valid & np.isfinite(x), x, 0.0)
        maximum = np.max(x, axis=1, keepdims=True)
        return np.divide(
            x,
            maximum,
            out=np.zeros_like(x),
            where=maximum > 0,
        )

    inverse = row_normalize(inverse)
    rank_signal = row_normalize(rank_signal)
    relative_distance = row_normalize(relative_distance)
    global_distance = row_normalize(global_distance)

    score = (
        np.power(inverse + eps, 0.40)
        * np.power(rank_signal + eps, 0.25)
        * np.power(relative_distance + eps, 0.20)
        * np.power(global_distance + eps, 0.15)
    )

    score *= 1.0 + 0.20 * mutual
    score = 0.5 * (score + score.T)
    score = np.where(valid, score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
