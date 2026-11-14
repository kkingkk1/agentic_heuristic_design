import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid & (distances > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    # Keep only valid undirected edges and use the shorter direction.
    d = np.where(valid, np.maximum(distances, 0.0), np.inf)
    d = np.minimum(d, d.T)
    valid = np.isfinite(d)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    fallback = (
        float(np.percentile(finite_values, 75.0))
        if finite_values.size
        else global_scale
    )
    fallback = max(fallback, global_scale)

    working = np.where(np.isfinite(d), d, 12.0 * fallback)

    positive_working = working[working > 0.0]
    local_fallback = (
        float(np.median(positive_working))
        if positive_working.size
        else global_scale
    )
    local_fallback = max(local_fallback, np.finfo(float).eps)

    # Robust endpoint-local scale from the nearest few neighbors.
    k = min(6, n - 1)
    nearest_values = np.partition(working, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, local_fallback * 1e-10)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :],
                   np.finfo(float).eps)
    )
    relative_distance = working / pair_scale

    distance_term = np.exp(-np.clip(relative_distance, 0.0, 60.0))

    # Bidirectional rank preference.
    order = np.argsort(working, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_temperature = max(2.0, 0.30 * n)
    rank_term = np.exp(-(ranks + ranks.T) / rank_temperature)

    # Mutual-nearest-neighbor affinity.
    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = int(nearest[i])
        if j != i and j < n and nearest[j] == i and np.isfinite(d[i, j]):
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    mutual_term = 1.0 + 0.45 * mutual

    # Smoothly suppress edges much longer than both endpoint-local scales.
    endpoint_ratio = d / pair_scale
    excess = np.maximum(endpoint_ratio - 1.0, 0.0)
    long_edge_gate = 1.0 / (1.0 + 1.25 * excess)

    score = (
        distance_term
        * (0.50 + 0.35 * rank_term)
        * mutual_term
        * (0.70 + 0.30 * long_edge_gate)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    # Symmetric degree balancing reduces node and hub dominance.
    row_strength = score.sum(axis=1)
    denominator = np.sqrt(
        np.maximum(row_strength[:, None] * row_strength[None, :], 0.0)
    )

    balanced = np.divide(
        score,
        denominator,
        out=np.zeros_like(score),
        where=denominator > 0.0,
    )

    maximum = np.max(balanced)
    if np.isfinite(maximum) and maximum > 0.0:
        balanced /= maximum

    balanced[~np.isfinite(balanced)] = 0.0
    np.fill_diagonal(balanced, 0.0)

    return balanced
