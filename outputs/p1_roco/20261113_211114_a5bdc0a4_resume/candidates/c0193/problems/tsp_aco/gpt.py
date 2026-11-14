import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diverse, sparse-graph-aware and long-term TSP edge priors."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(raw)
    np.fill_diagonal(finite, False)

    valid = finite | finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    clipped = np.maximum(np.nan_to_num(raw, nan=0.0), 0.0)

    distance = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T

    distance[both] = 0.5 * (
        clipped[both] + clipped.T[both]
    )
    distance[only_forward] = clipped[only_forward]
    distance[only_backward] = clipped[only_backward]
    distance[~valid] = 0.0
    np.fill_diagonal(distance, 0.0)

    valid_values = distance[valid]
    positive = valid_values[valid_values > 0.0]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    # Robust global range for preventing extreme edges from dominating.
    low = float(np.percentile(valid_values, 10.0))
    high = float(np.percentile(valid_values, 90.0))
    spread = max(
        high - low,
        0.5 * global_scale,
        np.finfo(float).eps,
    )

    degree = valid.sum(axis=1).astype(int)

    # Use a small sparse candidate graph. Small instances retain all edges.
    if n <= 8:
        candidate_width = n - 1
    else:
        candidate_width = min(
            n - 1,
            max(4, int(np.ceil(np.sqrt(float(n))) + 2)),
        )

    candidate = np.zeros((n, n), dtype=bool)
    rank = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(distance[i, neighbors], kind="stable")
        ]
        local_order = order[:candidate_width]
        candidate[i, local_order] = True
        rank[i, local_order] = np.arange(local_order.size)

    # Keep an edge if either endpoint regards it as a candidate.
    candidate |= candidate.T
    candidate &= valid

    # Node-adaptive local scale prevents dense and sparse regions from
    # receiving the same absolute temperature.
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size:
            local_values = distance[i, neighbors]
            positive_local = local_values[local_values > 0.0]
            if positive_local.size:
                local_scale[i] = float(
                    np.median(positive_local[: min(positive_local.size, 6)])
                )

    local_scale = np.maximum(local_scale, 0.25 * global_scale)

    # Multi-scale local affinity, evaluated only on the sparse candidate graph.
    affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(candidate[i])
        if neighbors.size == 0:
            continue

        temperature = max(
            0.75 * local_scale[i],
            0.20 * spread,
            np.finfo(float).eps,
        )

        distances = distance[i, neighbors]
        values = np.exp(
            -np.minimum(distances, high + 4.0 * global_scale)
            / temperature
        )

        # Stable rank term prevents near ties from being decided solely by
        # small numerical distance differences.
        rank_term = 1.0 / np.sqrt(1.0 + rank[i, neighbors])
        row_values = 0.78 * values + 0.22 * rank_term

        maximum = float(np.max(row_values))
        if np.isfinite(maximum) and maximum > 0.0:
            row_values /= maximum

        affinity[i, neighbors] = row_values

    affinity[~candidate] = 0.0

    # Symmetric local evidence retains the stronger endpoint signal while
    # preserving mutual plausibility.
    local = 0.58 * np.maximum(affinity, affinity.T)
    local += 0.42 * np.sqrt(affinity * affinity.T)
    local[~candidate] = 0.0

    # Candidate entropy estimates how constrained each node is. Low entropy
    # nodes receive protection, but high entropy nodes are not suppressed.
    probabilities = affinity.copy()
    row_sum = probabilities.sum(axis=1, keepdims=True)
    safe_row_sum = np.maximum(row_sum, np.finfo(float).eps)
    probabilities /= safe_row_sum

    entropy = np.zeros(n, dtype=float)

    for i in range(n):
        values = probabilities[i]
        positive_values = values[values > 0.0]
        if positive_values.size > 1:
            entropy[i] = -float(
                np.sum(positive_values * np.log(positive_values))
            ) / np.log(float(positive_values.size))

    scarcity = 1.0 - entropy
    scarcity = 0.72 + 0.48 * scarcity

    # Edge competition: an edge is globally useful when it is important to
    # at least one endpoint but does not monopolize both endpoints.
    endpoint_mass = np.maximum(affinity.sum(axis=1), np.finfo(float).eps)
    importance = affinity / endpoint_mass[:, None]

    exclusivity = np.maximum(importance, importance.T)
    sharedness = np.sqrt(
        np.maximum(importance, 0.0)
        * np.maximum(importance.T, 0.0)
    )

    diversity = (
        0.56 * exclusivity
        + 0.24 * np.sqrt(scarcity[:, None] * scarcity[None, :])
        + 0.20 * (1.0 - sharedness)
    )
    diversity = np.clip(diversity, 0.0, 1.0)
    diversity[~candidate] = 0.0

    # Sparse two-step continuation. For each candidate edge, estimate whether
    # both endpoints retain good alternatives after that edge is selected.
    continuation = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(candidate[i])
        if neighbors.size == 0:
            continue

        row_values = local[i, neighbors]
        if row_values.size > 1:
            alternatives = np.sort(row_values)[::-1]
            # The second-best and third-best alternatives summarize remaining
            # flexibility without constructing a dense two-hop matrix.
            if alternatives.size >= 3:
                level = 0.55 * alternatives[1] + 0.45 * alternatives[2]
            else:
                level = float(alternatives[1])
        else:
            level = 0.0

        continuation[i, neighbors] = level

    continuation_max = float(np.max(continuation))

    if np.isfinite(continuation_max) and continuation_max > 0.0:
        continuation /= continuation_max

    # Bridge-like edges are identified by whether their endpoints have
    # dissimilar candidate neighborhoods. This rewards useful global reach
    # without a full graph shortest-path computation.
    neighborhood_overlap = np.zeros((n, n), dtype=float)

    candidate_float = candidate.astype(float)
    candidate_degree = np.maximum(
        candidate_float.sum(axis=1),
        1.0,
    )

    for i in range(n):
        shared = candidate_float @ candidate_float[i]
        neighborhood_overlap[i] = shared / np.sqrt(
            candidate_degree * candidate_degree[i]
        )

    neighborhood_overlap = np.clip(neighborhood_overlap, 0.0, 1.0)
    cross_region = 1.0 - neighborhood_overlap

    long_term = (
        0.48 * np.sqrt(
            np.maximum(continuation, 0.0)
            * np.maximum(continuation.T, 0.0)
        )
        + 0.27 * cross_region
        + 0.25 * local
    )
    long_term[~candidate] = 0.0

    # Kruskal forest supplies a low-strength connectivity safeguard.
    forest = np.zeros((n, n), dtype=float)
    parent = np.arange(n, dtype=int)
    component_size = np.ones(n, dtype=int)

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

        if component_size[root_left] < component_size[root_right]:
            root_left, root_right = root_right, root_left

        parent[root_right] = root_left
        component_size[root_left] += component_size[root_right]
        return True

    upper_edges = np.argwhere(np.triu(candidate, k=1))

    if upper_edges.size:
        costs = distance[
            upper_edges[:, 0],
            upper_edges[:, 1],
        ]
        order = np.argsort(costs, kind="stable")

        for index in order:
            left = int(upper_edges[index, 0])
            right = int(upper_edges[index, 1])

            if union(left, right):
                forest[left, right] = 1.0
                forest[right, left] = 1.0

    combined = (
        0.52 * local
        + 0.29 * diversity * long_term
        + 0.14 * long_term
        + 0.05 * forest
    )
    combined[~candidate] = 0.0

    # Mildly protect scarce endpoints without allowing them to dominate.
    scarcity_edge = np.sqrt(scarcity[:, None] * scarcity[None, :])
    combined *= 0.86 + 0.14 * scarcity_edge
    combined[~valid] = 0.0

    # Final symmetric averaging is intentionally performed only once.
    combined = 0.5 * (combined + combined.T)
    combined[~valid] = 0.0
    np.fill_diagonal(combined, 0.0)

    maximum = float(np.max(combined))

    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    result = combined / maximum
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
