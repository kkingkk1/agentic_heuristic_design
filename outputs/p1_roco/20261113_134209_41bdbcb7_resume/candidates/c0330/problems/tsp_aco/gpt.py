import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    forward = np.where(raw_valid, distances, np.inf)
    backward = forward.T

    both_valid = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = np.isfinite(backward) & ~np.isfinite(forward)

    safe = np.full((n, n), np.inf, dtype=float)
    safe[both_valid] = 0.5 * (
        forward[both_valid] + backward[both_valid]
    )
    safe[only_forward] = forward[only_forward]
    safe[only_backward] = backward[only_backward]

    np.fill_diagonal(safe, np.inf)
    valid = np.isfinite(safe)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_valid = valid[upper_i, upper_j]
    edge_i = upper_i[upper_valid]
    edge_j = upper_j[upper_valid]
    edge_cost = safe[edge_i, edge_j]

    order = np.argsort(edge_cost, kind="stable")
    edge_i = edge_i[order]
    edge_j = edge_j[order]
    edge_cost = edge_cost[order]

    parent = np.arange(n, dtype=int)
    size = np.ones(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = np.zeros((n, n), dtype=bool)

    for i, j in zip(edge_i, edge_j):
        ri = find(int(i))
        rj = find(int(j))

        if ri == rj:
            continue

        if size[ri] < size[rj]:
            ri, rj = rj, ri

        parent[rj] = ri
        size[ri] += size[rj]
        tree[i, j] = True
        tree[j, i] = True

    tree_degree = tree.sum(axis=1).astype(float)

    tree_neighbors = [
        np.flatnonzero(tree[i]).tolist()
        for i in range(n)
    ]

    path_max = np.full((n, n), np.inf, dtype=float)

    for source in range(n):
        stack = [(source, -1, 0.0)]

        while stack:
            node, previous, current_max = stack.pop()

            if node != source:
                path_max[source, node] = current_max

            for neighbor in tree_neighbors[node]:
                if neighbor == previous:
                    continue

                edge_value = safe[node, neighbor]
                stack.append(
                    (
                        neighbor,
                        node,
                        max(current_max, float(edge_value)),
                    )
                )

    normalized_cost = np.clip(safe / scale, 0.0, 60.0)
    global_distance = np.exp(-0.85 * normalized_cost)

    exchange_gain = np.ones((n, n), dtype=float)
    non_tree = valid & ~tree

    exchange_delta = np.zeros((n, n), dtype=float)
    exchange_delta[non_tree] = (
        path_max[non_tree] - safe[non_tree]
    )

    exchange_gain[non_tree] = np.exp(
        np.clip(exchange_delta[non_tree] / scale, -4.0, 3.0)
    )
    exchange_gain[tree] = 1.12

    degree_need = np.exp(
        np.clip(0.22 * (2.0 - tree_degree), -0.8, 0.8)
    )
    degree_factor = np.sqrt(
        degree_need[:, None] * degree_need[None, :]
    )

    tree_degree_penalty = np.ones((n, n), dtype=float)
    overloaded = np.maximum(tree_degree - 2.0, 0.0)
    tree_degree_penalty *= np.exp(
        -0.10 * (overloaded[:, None] + overloaded[None, :])
    )

    non_tree_diversity = np.ones((n, n), dtype=float)
    non_tree_diversity[non_tree] = 1.0 + 0.06 * np.clip(
        normalized_cost[non_tree],
        0.0,
        3.0,
    )

    scores = (
        global_distance
        * exchange_gain
        * degree_factor
        * tree_degree_penalty
        * non_tree_diversity
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    target_degree = np.minimum(
        2.0,
        valid.sum(axis=1).astype(float),
    )

    scaling = np.ones(n, dtype=float)

    for _ in range(32):
        row_sum = scores @ scaling
        update = np.divide(
            target_degree,
            np.maximum(row_sum, eps),
        )
        scaling *= np.sqrt(np.clip(update, 0.05, 20.0))
        scaling = np.clip(scaling, 1e-4, 1e4)

    scores *= scaling[:, None] * scaling[None, :]
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
