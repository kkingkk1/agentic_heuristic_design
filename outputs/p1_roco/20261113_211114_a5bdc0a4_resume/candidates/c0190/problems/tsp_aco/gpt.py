import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse and long-term-aware edge priors for TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    np.fill_diagonal(finite, False)

    valid = finite | finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clipped = np.maximum(np.nan_to_num(raw, nan=0.0), 0.0)

    # Symmetrize correctly:
    # - two-sided valid edges use their mean;
    # - one-sided valid edges copy the existing valid direction directly.
    distance = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    distance[both] = 0.5 * (
        clipped[both] + clipped.T[both]
    )
    distance[forward_only] = clipped[forward_only]
    distance[backward_only] = clipped[backward_only]

    distance[~valid] = 0.0
    np.fill_diagonal(distance, 0.0)

    values = distance[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    low = float(np.percentile(values, 10.0))
    high = float(np.percentile(values, 90.0))
    spread = max(high - low, 0.35 * scale, np.finfo(float).eps)

    # Stable row-wise ranks and local option counts.
    rank = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(distance[i, neighbors], kind="stable")
        ]
        rank[i, order] = np.arange(order.size, dtype=float)

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / np.sqrt(1.0 + rank[valid])

    # Mutual-neighbor structure favors edges that are locally plausible
    # from both endpoints, while retaining non-mutual alternatives.
    mutual_rank = np.zeros((n, n), dtype=float)
    mutual_rank[valid] = np.sqrt(
        rank_score[valid] * rank_score.T[valid]
    )

    # Multi-scale affinity.
    temperatures = np.array(
        [0.35 * spread, 0.85 * spread, 1.8 * spread, 3.5 * spread],
        dtype=float,
    )
    temperatures = np.maximum(temperatures, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)
    capped_distance = np.minimum(distance, high + 10.0 * scale)

    for temperature in temperatures:
        component = np.exp(-capped_distance / temperature)
        component[~valid] = 0.0
        np.fill_diagonal(component, 0.0)

        maximum = float(np.max(component))
        if np.isfinite(maximum) and maximum > 0.0:
            component /= maximum

        affinity += component

    affinity /= float(temperatures.size)

    # Blend metric affinity with rank affinity.
    local = 0.70 * affinity + 0.30 * mutual_rank
    local[~valid] = 0.0

    # Node-level scarcity: low total affinity means fewer attractive choices.
    row_mass = local.sum(axis=1)
    positive_mass = row_mass[row_mass > 0.0]

    reference_mass = (
        float(np.median(positive_mass))
        if positive_mass.size
        else 1.0
    )
    reference_mass = max(reference_mass, np.finfo(float).eps)

    scarcity = np.sqrt(
        reference_mass / np.maximum(row_mass, np.finfo(float).eps)
    )
    scarcity = np.clip(scarcity, 0.65, 1.85)

    scarcity_edge = np.sqrt(scarcity[:, None] * scarcity[None, :])
    scarcity_edge /= max(float(np.max(scarcity_edge)), 1.0)

    # Endpoint exclusivity: reward an edge when it is relatively important
    # to at least one endpoint, without forcing mutual nearest neighbors.
    valid_degree = valid.sum(axis=1).astype(float)
    degree_factor = 1.0 / np.sqrt(np.maximum(valid_degree, 1.0))
    degree_edge = np.sqrt(degree_factor[:, None] * degree_factor[None, :])
    degree_edge /= max(float(np.max(degree_edge)), 1.0)

    exclusivity = 0.58 * mutual_rank + 0.42 * degree_edge
    exclusivity[~valid] = 0.0

    diverse = local * (
        0.62
        + 0.23 * scarcity_edge
        + 0.15 * exclusivity
    )
    diverse[~valid] = 0.0

    # Row-normalized transition prior for long-term continuation.
    transition = affinity.copy()
    transition_sum = transition.sum(axis=1, keepdims=True)
    transition_sum = np.maximum(
        transition_sum,
        np.finfo(float).eps,
    )
    transition /= transition_sum

    # Two-hop support measures whether an edge participates in plausible
    # longer structures rather than only being a direct nearest neighbor.
    two_hop = transition @ transition
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop[~valid] = 0.0

    two_hop_max = float(np.max(two_hop))
    if np.isfinite(two_hop_max) and two_hop_max > 0.0:
        two_hop /= two_hop_max

    # Best two-edge detour. A large detour advantage indicates that
    # removing the edge may damage long-range connectivity.
    best_detour = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        candidate = distance[:, [k]] + distance[[k], :]
        candidate_valid = valid[:, [k]] & valid[[k], :]
        candidate[~candidate_valid] = np.inf
        best_detour = np.minimum(best_detour, candidate)

    advantage = np.maximum(best_detour - distance, 0.0)
    advantage[~valid] = 0.0

    bridge_score = 1.0 - np.exp(
        -advantage / max(spread, scale)
    )
    bridge_score[~valid] = 0.0

    # Remaining-degree compatibility: an edge is useful when both endpoints
    # still retain several good continuations after selecting it.
    remaining_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        sorted_values = np.sort(local[i, neighbors])[::-1]
        if sorted_values.size >= 3:
            continuation_level = float(
                np.mean(sorted_values[1:4])
            )
        else:
            continuation_level = float(
                np.mean(sorted_values)
            )

        remaining_support[i, neighbors] = continuation_level

    remaining_support = np.sqrt(
        remaining_support * remaining_support.T
    )
    remaining_max = float(np.max(remaining_support))

    if np.isfinite(remaining_max) and remaining_max > 0.0:
        remaining_support /= remaining_max

    long_term = (
        0.34 * two_hop
        + 0.30 * bridge_score
        + 0.20 * remaining_support
        + 0.16 * local
    )
    long_term[~valid] = 0.0

    # True minimum spanning forest using Kruskal and union-find.
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
        edge_costs = distance[
            upper_edges[:, 0],
            upper_edges[:, 1],
        ]
        order = np.argsort(edge_costs, kind="stable")

        for index in order:
            left = int(upper_edges[index, 0])
            right = int(upper_edges[index, 1])

            if union(left, right):
                forest[left, right] = 1.0
                forest[right, left] = 1.0

    # Forest edges receive protection, but never dominate all other evidence.
    combined = (
        0.50 * diverse
        + 0.35 * long_term
        + 0.15 * forest
    )
    combined[~valid] = 0.0
    combined = 0.5 * (combined + combined.T)

    # Mild degree correction. This is deliberately blended with the original
    # scores to avoid the over-smoothing observed in repeated balancing.
    corrected = combined.copy()

    for _ in range(4):
        row_sum = corrected.sum(axis=1)
        row_scale = np.sqrt(
            np.maximum(row_sum, np.finfo(float).eps)
        )

        corrected = corrected / (
            row_scale[:, None] * row_scale[None, :]
        )
        corrected[~valid] = 0.0
        corrected = 0.5 * (corrected + corrected.T)

        maximum = float(np.max(corrected))
        if np.isfinite(maximum) and maximum > 0.0:
            corrected /= maximum

    result = 0.76 * combined + 0.24 * corrected
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
