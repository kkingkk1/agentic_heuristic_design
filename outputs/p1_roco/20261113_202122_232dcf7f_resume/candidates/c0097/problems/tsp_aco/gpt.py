import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return cycle-aware edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0.0):
        raise ValueError("distance_matrix must be nonnegative")

    d = 0.5 * (distances + distances.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Build a minimum spanning tree with Prim's algorithm.
    tree = np.zeros((n, n), dtype=bool)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        u = int(np.argmin(candidates))

        if not np.isfinite(candidates[u]):
            break

        in_tree[u] = True
        p = parent[u]
        if p >= 0:
            tree[u, p] = True
            tree[p, u] = True

        unvisited = ~in_tree
        improved = unvisited & (d[u] < best)
        best[improved] = d[u, improved]
        parent[improved] = u

    tree_degree = tree.sum(axis=1).astype(float)

    # Compute the largest tree edge on every tree path.
    path_max = np.zeros((n, n), dtype=float)

    for source in range(n):
        visited = np.zeros(n, dtype=bool)
        stack = [(source, -1, 0.0)]

        while stack:
            node, previous, current_max = stack.pop()
            if visited[node]:
                continue

            visited[node] = True
            path_max[source, node] = current_max

            neighbors = np.flatnonzero(tree[node])
            for neighbor in neighbors:
                if neighbor != previous and not visited[neighbor]:
                    next_max = max(current_max, d[node, neighbor])
                    stack.append((int(neighbor), node, next_max))

    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)

    # Direct scale-aware closeness.
    closeness = np.exp(-d / scale)
    closeness[~valid] = 0.0

    # Edges that do not exceed the bottleneck of their tree path are favored.
    excess = np.maximum(d - path_max, 0.0)
    cycle_closure = np.exp(-excess / scale)
    cycle_closure[~valid] = 0.0

    # Preserve explicit spanning-tree structure.
    backbone = tree.astype(float)

    # Prefer edges that help low-degree endpoints without dominating the score.
    degree_pressure = 1.0 / np.sqrt(
        (1.0 + tree_degree[:, None]) * (1.0 + tree_degree[None, :])
    )
    degree_pressure /= max(float(np.max(degree_pressure)), 1.0)
    degree_pressure[~valid] = 0.0

    score = (
        0.45 * backbone
        + 0.30 * cycle_closure
        + 0.15 * closeness
        + 0.10 * degree_pressure
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
