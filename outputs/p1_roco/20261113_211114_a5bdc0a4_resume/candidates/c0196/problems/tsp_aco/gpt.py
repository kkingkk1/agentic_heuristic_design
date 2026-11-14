import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return exploration-aware edge priors for stochastic TSP sampling."""
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

    clipped = np.maximum(np.nan_to_num(raw, nan=0.0), 0.0)

    # Undirected structural costs are used only for topology-sensitive
    # operations. Directional evidence is retained separately below.
    distance = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    distance[both] = 0.5 * (clipped[both] + clipped.T[both])
    distance[forward_only] = clipped[forward_only]
    distance[backward_only] = clipped.T[backward_only]
    distance[~valid] = 0.0
    np.fill_diagonal(distance, 0.0)

    observed = distance[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    low = float(np.percentile(observed, 10.0))
    high = float(np.percentile(observed, 90.0))
    spread = max(high - low, 0.5 * scale, np.finfo(float).eps)

    # Sparse candidate graph: retain short local edges while keeping
    # asymmetric one-sided edges eligible through union symmetrization.
    if n <= 8:
        width = n - 1
    else:
        width = min(n - 1, max(4, int(np.ceil(np.sqrt(float(n))) + 2)))

    candidate = np.zeros((n, n), dtype=bool)
    rank = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        order = neighbors[np.argsort(distance[i, neighbors], kind="stable")]
        chosen = order[:width]
        candidate[i, chosen] = True
        rank[i, chosen] = np.arange(chosen.size, dtype=float)

    candidate |= candidate.T
    candidate &= valid

    # Directional affinity preserves stronger evidence from either input
    # orientation instead of averaging it away at the start.
    directional = np.zeros((n, n), dtype=float)
    raw_positive = np.maximum(clipped, 0.0)
    directional[finite] = np.exp(
        -np.minimum(raw_positive[finite], high + 4.0 * scale)
        / max(0.45 * scale, np.finfo(float).eps)
    )

    # Fill missing orientations with the structural edge evidence.
    missing = valid & ~finite
    directional[missing] = np.exp(
        -np.minimum(distance[missing], high + 4.0 * scale)
        / max(0.45 * scale, np.finfo(float).eps)
    )

    local_scale = np.full(n, scale, dtype=float)
    nearest = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)
    nearest_index = np.full(n, -1, dtype=int)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        values = distance[i, neighbors]
        order = np.argsort(values, kind="stable")
        ordered_neighbors = neighbors[order]
        ordered_values = values[order]

        nearest[i] = ordered_values[0]
        nearest_index[i] = int(ordered_neighbors[0])

        if ordered_values.size > 1:
            second[i] = ordered_values[1]
            local_scale[i] = max(
                ordered_values[1] - ordered_values[0],
                0.20 * scale,
            )
        else:
            local_scale[i] = 0.20 * scale

    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Endpoint exclusivity rewards edges that are especially important to
    # at least one endpoint, while remaining bounded.
    endpoint = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        reference = second[i] if np.isfinite(second[i]) else nearest[i]
        values = np.exp(
            np.clip(
                (reference - distance[i, neighbors]) / local_scale[i],
                -12.0,
                12.0,
            )
        )
        maximum = float(np.max(values))
        if maximum > 0.0 and np.isfinite(maximum):
            values /= maximum
        endpoint[i, neighbors] = values

    endpoint_edge = np.sqrt(endpoint * endpoint.T)
    endpoint_edge[~valid] = 0.0

    # Mutual-nearest-neighbour protection.
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_index[i]
        if j >= 0 and nearest_index[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Local affinity combines directional evidence with stable rank.
    affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(candidate[i])
        if neighbors.size == 0:
            continue

        values = directional[i, neighbors]
        rank_term = 1.0 / np.sqrt(1.0 + rank[i, neighbors])
        row = 0.78 * values + 0.22 * rank_term

        maximum = float(np.max(row))
        if maximum > 0.0 and np.isfinite(maximum):
            row /= maximum

        affinity[i, neighbors] = row

    affinity_edge = np.maximum(affinity, affinity.T)
    affinity_edge[~candidate] = 0.0

    # Candidate entropy supplies a mild scarcity signal.
    row_sum = affinity.sum(axis=1, keepdims=True)
    probabilities = affinity / np.maximum(row_sum, np.finfo(float).eps)
    entropy = np.zeros(n, dtype=float)

    for i in range(n):
        values = probabilities[i]
        positive_values = values[values > 0.0]
        if positive_values.size > 1:
            entropy[i] = -float(
                np.sum(positive_values * np.log(positive_values))
            ) / np.log(float(positive_values.size))

    scarcity = 0.72 + 0.48 * (1.0 - entropy)
    scarcity_edge = np.sqrt(scarcity[:, None] * scarcity[None, :])

    # Restricted two-hop bypass estimate.
    k = min(8, n - 1)
    nearest_indices = np.full((n, k), -1, dtype=int)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(distance[i, neighbors], kind="stable")
        ]
        take = min(k, order.size)
        nearest_indices[i, :take] = order[:take]

    best_bypass = np.full((n, n), np.inf, dtype=float)

    for column in range(k):
        middles = nearest_indices[:, column]
        for middle in middles[middles >= 0]:
            feasible = valid[:, middle, None] & valid[middle, None, :]
            route = (
                distance[:, middle, None]
                + distance[None, middle, :]
            )
            best_bypass = np.minimum(
                best_bypass,
                np.where(feasible, route, np.inf),
            )

    measurable = candidate & np.isfinite(best_bypass)
    gain = np.zeros((n, n), dtype=float)
    gain[measurable] = np.maximum(
        best_bypass[measurable] - distance[measurable],
        0.0,
    )

    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[measurable] = 1.0 - np.exp(
        -gain[measurable] / max(0.85 * scale, np.finfo(float).eps)
    )

    # Neighbourhood dissimilarity encourages useful exploratory bridges.
    candidate_float = candidate.astype(float)
    degree = np.maximum(candidate_float.sum(axis=1), 1.0)
    overlap = np.zeros((n, n), dtype=float)

    for i in range(n):
        shared = candidate_float @ candidate_float[i]
        overlap[i] = shared / np.sqrt(degree * degree[i])

    cross_region = np.clip(1.0 - overlap, 0.0, 1.0)
    cross_region[~candidate] = 0.0

    # Low-strength Kruskal forest safeguard for sparse connectivity.
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

    upper = np.argwhere(np.triu(candidate, k=1))
    if upper.size:
        costs = distance[upper[:, 0], upper[:, 1]]
        order = np.argsort(costs, kind="stable")

        for position in order:
            left = int(upper[position, 0])
            right = int(upper[position, 1])
            if union(left, right):
                forest[left, right] = 1.0
                forest[right, left] = 1.0

    # Exploitation: local affinity, endpoint scarcity, and replacement cost.
    exploit = (
        0.52 * affinity_edge
        + 0.24 * endpoint_edge
        + 0.14 * irreplaceability
        + 0.10 * mutual
    )

    # Exploration: bridge structure and sparse continuation flexibility.
    continuation = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(candidate[i])
        values = np.sort(affinity[i, neighbors])[::-1]

        if values.size >= 3:
            level = 0.55 * values[1] + 0.45 * values[2]
        elif values.size == 2:
            level = values[1]
        else:
            level = 0.0

        continuation[i, neighbors] = level

    continuation_max = float(np.max(continuation))
    if continuation_max > 0.0 and np.isfinite(continuation_max):
        continuation /= continuation_max

    continuation_edge = np.sqrt(
        np.maximum(continuation, 0.0)
        * np.maximum(continuation.T, 0.0)
    )

    explore = (
        0.42 * continuation_edge
        + 0.33 * cross_region
        + 0.15 * endpoint_edge
        + 0.10 * forest
    )

    score = (
        0.68 * exploit
        + 0.24 * explore
        + 0.08 * irreplaceability
    )

    score *= 0.86 + 0.14 * scarcity_edge
    score[~candidate] = 0.0

    # Preserve the stronger directional evidence for each undirected edge.
    score = np.maximum(score, score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    result = score / maximum
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
