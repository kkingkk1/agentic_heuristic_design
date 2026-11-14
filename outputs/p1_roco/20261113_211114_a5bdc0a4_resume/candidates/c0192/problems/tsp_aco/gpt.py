import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced local and long-term edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & offdiag
    valid = finite | finite.T

    if not np.any(valid):
        return result

    nonnegative = np.maximum(np.nan_to_num(raw, nan=0.0, posinf=0.0), 0.0)
    distances = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T

    distances[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )
    distances[only_forward] = nonnegative[only_forward]
    distances[only_backward] = nonnegative.T[only_backward]

    distances[~valid] = 0.0
    np.fill_diagonal(distances, 0.0)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps

    low = float(np.percentile(observed, 10.0))
    high = float(np.percentile(observed, 90.0))
    spread = max(high - low, 0.35 * scale, eps)

    # Multi-scale distance affinity.
    temperatures = np.maximum(
        np.array([0.45, 1.10, 2.40]) * spread,
        eps,
    )
    affinity = np.zeros((n, n), dtype=float)
    capped = np.minimum(distances, high + 8.0 * scale)

    for temperature in temperatures:
        component = np.exp(-capped / temperature)
        component[~valid] = 0.0
        affinity += component

    affinity /= float(temperatures.size)
    affinity_max = float(np.max(affinity))

    if np.isfinite(affinity_max) and affinity_max > 0.0:
        affinity /= affinity_max

    # Stable rank and mutual-neighborhood affinity.
    rank_score = np.zeros((n, n), dtype=float)
    nearest_index = np.full(n, -1, dtype=int)
    candidate_lists = []

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])

        if neighbors.size == 0:
            candidate_lists.append(np.empty(0, dtype=int))
            continue

        order = neighbors[
            np.argsort(distances[i, neighbors], kind="stable")
        ]
        rank_score[i, order] = 1.0 / np.sqrt(
            1.0 + np.arange(order.size, dtype=float)
        )
        nearest_index[i] = int(order[0])
        candidate_lists.append(order[: min(8, order.size)])

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_rank[~valid] = 0.0

    # Endpoint exclusivity based on the gap between the first two choices.
    nearest_distance = np.full(n, np.inf, dtype=float)
    second_distance = np.full(n, np.inf, dtype=float)

    for i, neighbors in enumerate(candidate_lists):
        if neighbors.size:
            nearest_distance[i] = distances[i, neighbors[0]]
        if neighbors.size > 1:
            second_distance[i] = distances[i, neighbors[1]]

    endpoint = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        if np.isfinite(second_distance[i]):
            reference = second_distance[i]
        else:
            reference = nearest_distance[i]

        local_scale = max(
            abs(reference - nearest_distance[i]),
            0.20 * scale,
            eps,
        )

        values = np.exp(
            np.clip(
                (reference - distances[i, neighbors]) / local_scale,
                -12.0,
                12.0,
            )
        )
        endpoint[i, neighbors] = values

    endpoint = np.sqrt(endpoint * endpoint.T)
    endpoint[~valid] = 0.0

    mutual_nearest = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest_index):
        if j >= 0 and nearest_index[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    # Main local score. This retains the exploiter's strong short-term bias.
    local = (
        0.68 * affinity
        + 0.20 * endpoint
        + 0.08 * mutual_rank
        + 0.04 * mutual_nearest
    )
    local[~valid] = 0.0

    # Restricted two-hop replacement signal.
    best_bypass = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        left_valid = valid[:, middle]
        right_valid = valid[middle, :]

        if not np.any(left_valid) or not np.any(right_valid):
            continue

        route = distances[:, middle, None] + distances[middle, :][None, :]
        feasible = left_valid[:, None] & right_valid[None, :]
        candidate = np.where(feasible, route, np.inf)
        best_bypass = np.minimum(best_bypass, candidate)

    measurable = valid & np.isfinite(best_bypass)
    gain = np.zeros((n, n), dtype=float)
    gain[measurable] = np.maximum(
        best_bypass[measurable] - distances[measurable],
        0.0,
    )

    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[measurable] = 1.0 - np.exp(
        -gain[measurable] / max(0.85 * scale, eps)
    )

    # Row-normalized transition support for plausible two-hop structures.
    transition = affinity.copy()
    row_sum = transition.sum(axis=1, keepdims=True)
    transition /= np.maximum(row_sum, eps)

    two_hop = transition @ transition
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop[~valid] = 0.0

    two_hop_max = float(np.max(two_hop))

    if np.isfinite(two_hop_max) and two_hop_max > 0.0:
        two_hop /= two_hop_max

    # Kruskal forest protects a sparse low-cost connectivity skeleton.
    forest = np.zeros((n, n), dtype=float)
    parent = np.arange(n, dtype=int)
    size = np.ones(n, dtype=int)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]

        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node

        return root

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)

        if root_left == root_right:
            return False

        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left

        parent[root_right] = root_left
        size[root_left] += size[root_right]
        return True

    upper_edges = np.argwhere(np.triu(valid, k=1))

    if upper_edges.size:
        costs = distances[upper_edges[:, 0], upper_edges[:, 1]]
        order = np.argsort(costs, kind="stable")

        for index in order:
            left = int(upper_edges[index, 0])
            right = int(upper_edges[index, 1])

            if union(left, right):
                forest[left, right] = 1.0
                forest[right, left] = 1.0

    # Long-term structure is deliberately subordinate to local evidence.
    long_term = (
        0.38 * two_hop
        + 0.34 * irreplaceability
        + 0.18 * forest
        + 0.10 * local
    )
    long_term[~valid] = 0.0

    # Conservative exploration-exploitation integration.
    score = (
        0.78 * local
        + 0.14 * long_term
        + 0.08 * affinity
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))

    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
