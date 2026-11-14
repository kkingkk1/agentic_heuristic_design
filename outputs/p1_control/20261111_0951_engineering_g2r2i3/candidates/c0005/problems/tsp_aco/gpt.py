import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-exchange and degree-balancing edge indicators for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)
    tree = np.zeros((n, n), dtype=bool)
    adjacency = [[] for _ in range(n)]
    visited = np.zeros(n, dtype=bool)

    # Build a minimum spanning forest with Prim's algorithm.
    for root in range(n):
        if visited[root]:
            continue

        visited[root] = True
        best_cost = np.full(n, np.inf, dtype=float)
        best_parent = np.full(n, -1, dtype=int)
        best_cost[:] = safe[root]
        best_cost[visited] = np.inf

        while True:
            child = int(np.argmin(best_cost))
            if not np.isfinite(best_cost[child]):
                break

            parent = int(best_parent[child])
            if parent < 0:
                break

            tree[parent, child] = True
            tree[child, parent] = True
            adjacency[parent].append((child, float(safe[parent, child])))
            adjacency[child].append((parent, float(safe[parent, child])))

            visited[child] = True
            best_cost[child] = np.inf

            for candidate in range(n):
                if not visited[candidate] and safe[child, candidate] < best_cost[candidate]:
                    best_cost[candidate] = safe[child, candidate]
                    best_parent[candidate] = child

    tree_degree = tree.sum(axis=1).astype(float)
    degree_deficit = np.maximum(0.0, 2.0 - tree_degree)
    deficit_pair = degree_deficit[:, None] * degree_deficit[None, :]
    if np.max(deficit_pair) > 0.0:
        deficit_pair /= np.max(deficit_pair)

    # Compute the maximum tree-edge cost on every connected tree path.
    path_max = np.full((n, n), np.nan, dtype=float)
    for source in range(n):
        path_max[source, source] = 0.0
        stack = [(source, -1, 0.0)]

        while stack:
            node, parent, current_max = stack.pop()
            for neighbor, edge_cost in adjacency[node]:
                if neighbor == parent:
                    continue
                next_max = max(current_max, edge_cost)
                path_max[source, neighbor] = next_max
                stack.append((neighbor, node, next_max))

    # Potential gain from replacing the largest tree edge on a fundamental cycle.
    exchange_gain = np.zeros((n, n), dtype=float)
    exchange_possible = np.isfinite(path_max) & (~tree) & valid
    exchange_gain[exchange_possible] = np.maximum(
        path_max[exchange_possible] - d[exchange_possible],
        0.0,
    )
    if np.max(exchange_gain) > 0.0:
        exchange_gain /= np.max(exchange_gain)

    # Bounded global distance affinity; no row-wise normalization is used.
    distance_affinity = 1.0 / (1.0 + np.square(np.maximum(d, 0.0) / scale))
    distance_affinity[~valid] = 0.0

    tree_bonus = tree.astype(float)
    score = (
        0.20 * distance_affinity
        + 0.50 * tree_bonus
        + 0.20 * deficit_pair
        + 0.10 * exchange_gain
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        np.maximum(score, 0.0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
