import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return topology-aware edge-priority scores for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.zeros_like(d, dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric[valid]
    global_scale = max(float(np.median(positive_values)), eps)

    work = np.where(valid, symmetric, np.inf)
    np.fill_diagonal(work, np.inf)

    # Global direct-edge affinity.
    direct_affinity = np.exp(
        -np.minimum(work / global_scale, 60.0)
    )
    direct_affinity = np.where(valid, direct_affinity, 0.0)

    # Bilateral soft competition: an edge is preferred when both endpoints
    # assign it substantial probability mass.
    finite_count = np.sum(valid, axis=1).astype(float)
    row_mean = np.divide(
        np.sum(np.where(valid, symmetric, 0.0), axis=1),
        np.maximum(finite_count, 1.0),
    )
    row_temperature = np.maximum(row_mean, global_scale * 0.05)

    with np.errstate(over="ignore", invalid="ignore"):
        soft = np.exp(
            -np.minimum(work / row_temperature[:, None], 60.0)
        )
    soft = np.where(valid, soft, 0.0)

    row_mass = np.sum(soft, axis=1)
    row_probability = soft / np.maximum(row_mass[:, None], eps)
    bilateral_competition = np.sqrt(
        row_probability * row_probability.T
    )
    bilateral_competition = np.where(valid, bilateral_competition, 0.0)

    # Build a minimum spanning forest with Kruskal's algorithm.
    upper_i, upper_j = np.where(np.triu(valid, k=1))
    edge_order = np.argsort(
        symmetric[upper_i, upper_j],
        kind="stable",
    )

    parent = np.arange(n, dtype=int)
    size = np.ones(n, dtype=int)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    spanning = np.zeros((n, n), dtype=float)

    for position in edge_order:
        left = int(upper_i[position])
        right = int(upper_j[position])

        root_left = find(left)
        root_right = find(right)

        if root_left == root_right:
            continue

        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left

        parent[root_right] = root_left
        size[root_left] += size[root_right]
        spanning[left, right] = 1.0
        spanning[right, left] = 1.0

    # Estimate the shortest available two-hop detour for every pair.
    # This is evaluated without allocating an n-by-n-by-n tensor.
    two_hop = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        left_hops = work[:, middle][:, None]
        right_hops = work[middle, :][None, :]
        two_hop = np.minimum(two_hop, left_hops + right_hops)

    finite_detour = np.isfinite(two_hop) & valid
    bridge_score = np.zeros((n, n), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        detour_gap = (two_hop - symmetric) / (
            two_hop + symmetric + eps
        )

    bridge_score[finite_detour] = np.clip(
        detour_gap[finite_detour],
        0.0,
        1.0,
    )

    # Additive fusion keeps every valid edge available while emphasizing
    # globally useful backbone and bridge edges.
    score = (
        0.35 * direct_affinity
        + 0.25 * bilateral_competition
        + 0.25 * spanning
        + 0.15 * bridge_score
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
