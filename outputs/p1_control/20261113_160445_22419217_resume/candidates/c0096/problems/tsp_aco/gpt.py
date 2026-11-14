import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric MST-cycle-exchange prior for undirected TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    symmetric = 0.5 * (distances + distances.T)
    symmetric[~valid] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    finite_edges = symmetric[valid]
    positive_edges = finite_edges[finite_edges > 0.0]
    if positive_edges.size:
        scale = float(np.median(positive_edges))
    elif finite_edges.size:
        scale = float(np.median(finite_edges))
    else:
        scale = 1.0

    eps = np.finfo(float).eps
    scale = max(scale, eps)

    # Build a minimum spanning forest with Prim's algorithm.
    tree = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for _ in range(n):
        unvisited = np.flatnonzero(~visited)
        if unvisited.size == 0:
            break

        current_keys = key[unvisited]
        finite_positions = np.flatnonzero(np.isfinite(current_keys))

        if finite_positions.size:
            current = int(unvisited[finite_positions[np.argmin(
                current_keys[finite_positions]
            )]])
        else:
            current = int(unvisited[0])
            key[current] = 0.0

        visited[current] = True
        predecessor = int(parent[current])

        if predecessor >= 0 and np.isfinite(symmetric[current, predecessor]):
            tree[current, predecessor] = True
            tree[predecessor, current] = True

        candidates = (~visited) & np.isfinite(symmetric[current])
        improved = candidates & (symmetric[current] < key)
        key[improved] = symmetric[current, improved]
        parent[improved] = current

    # Compute the maximum edge weight on each tree path.
    bottleneck = np.full((n, n), np.inf, dtype=float)

    for root in range(n):
        bottleneck[root, root] = 0.0
        stack = [(root, -1, 0.0)]

        while stack:
            node, predecessor, current_max = stack.pop()
            neighbors = np.flatnonzero(tree[node])

            for neighbor in neighbors:
                if neighbor == predecessor:
                    continue

                next_max = max(current_max, symmetric[node, neighbor])
                if not np.isfinite(bottleneck[root, neighbor]):
                    bottleneck[root, neighbor] = next_max
                    stack.append((int(neighbor), node, next_max))

    # For disconnected valid graphs, use a conservative global fallback.
    finite_bottlenecks = bottleneck[np.isfinite(bottleneck) & (bottleneck > 0.0)]
    fallback = (
        float(np.median(finite_bottlenecks))
        if finite_bottlenecks.size
        else scale
    )
    bottleneck = np.where(np.isfinite(bottleneck), bottleneck, fallback)

    # Bounded global short-edge preference.
    short_edge = scale / (scale + symmetric)

    # Positive gain from replacing the bottleneck on the tree path.
    improvement = np.maximum(bottleneck - symmetric, 0.0)
    cycle_gain = improvement / (scale + improvement)

    # Preserve useful tree structure, while keeping all valid edges available.
    tree_support = tree.astype(float) * short_edge

    score = (
        0.35 * short_edge
        + 0.40 * cycle_gain
        + 0.25 * tree_support
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
