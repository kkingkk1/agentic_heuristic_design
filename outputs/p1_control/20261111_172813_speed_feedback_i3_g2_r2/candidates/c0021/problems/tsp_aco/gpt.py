import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]

    if positive.size == 0:
        return result

    scale = float(np.median(positive))
    large = max(scale, 1.0) * 1e6

    safe_d = np.where(finite, np.maximum(d, 0.0), large)
    np.fill_diagonal(safe_d, np.inf)

    valid = finite & (~np.eye(n, dtype=bool))

    symmetric_d = 0.5 * (safe_d + safe_d.T)
    symmetric_d[~np.isfinite(symmetric_d)] = large
    np.fill_diagonal(symmetric_d, np.inf)

    # Scale-free inverse-distance preference.
    inv = np.zeros((n, n), dtype=float)
    inv[valid] = 1.0 / (safe_d[valid] + 1e-12)
    if np.max(inv) > 0:
        inv /= np.max(inv)

    # Rank-based directed neighborhood preference.
    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n)[None, :]

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid].astype(float))

    # Smooth mutual-neighborhood compatibility.
    mutual = np.sqrt(rank_score * rank_score.T)
    mutual[~valid] = 0.0

    # Shared-neighbor compatibility captures local structural agreement.
    k = min(3, n - 1)
    neighbor_mask = np.zeros((n, n), dtype=float)
    for i in range(n):
        neighbors = order[i, :k]
        neighbor_mask[i, neighbors] = 1.0
    np.fill_diagonal(neighbor_mask, 0.0)

    shared = neighbor_mask @ neighbor_mask.T
    shared /= float(max(k, 1))
    shared[~valid] = 0.0

    compatibility = 0.7 * mutual + 0.3 * shared

    # Local savings relative to the best alternative connection at each endpoint.
    two_best = np.partition(symmetric_d, 1, axis=1)[:, :2]
    nearest = np.argmin(symmetric_d, axis=1)
    nearest_distance = two_best[:, 0]
    second_distance = two_best[:, 1]

    alternative_i = np.broadcast_to(nearest_distance[:, None], (n, n)).copy()
    alternative_j = np.broadcast_to(nearest_distance[None, :], (n, n)).copy()

    nearest_is_j = nearest[:, None] == np.arange(n)[None, :]
    nearest_is_i = nearest[None, :] == np.arange(n)[:, None]

    alternative_i[nearest_is_j] = np.broadcast_to(
        second_distance[:, None], (n, n)
    )[nearest_is_j]
    alternative_j[nearest_is_i] = np.broadcast_to(
        second_distance[None, :], (n, n)
    )[nearest_is_i]

    savings = alternative_i + alternative_j - 2.0 * symmetric_d
    savings = np.maximum(savings, 0.0)
    savings[~valid] = 0.0

    max_savings = np.max(savings)
    if max_savings > 0:
        savings /= max_savings

    reciprocal = 0.5 * (inv + inv.T)

    score = (
        0.42 * inv
        + 0.18 * reciprocal
        + 0.25 * compatibility
        + 0.15 * savings
    )

    score[~valid] = 0.0

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
