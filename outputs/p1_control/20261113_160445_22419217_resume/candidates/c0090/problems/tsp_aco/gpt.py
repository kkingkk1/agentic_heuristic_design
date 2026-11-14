import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric TSP edge-priority matrix using MST alpha-nearness."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    # TSP is undirected here; remove any directional inconsistency.
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    distance_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    distance_scale = max(distance_scale, np.finfo(float).eps)

    # Prim minimum spanning tree.
    parent = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf, dtype=float)
    used = np.zeros(n, dtype=bool)
    best[0] = 0.0

    for _ in range(n):
        available = np.where(~used)[0]
        current = available[np.argmin(best[available])]
        used[current] = True

        neighbors = np.where(~used)[0]
        if neighbors.size:
            improved = d[current, neighbors] < best[neighbors]
            changed = neighbors[improved]
            best[changed] = d[current, changed]
            parent[changed] = current

    tree = np.zeros((n, n), dtype=bool)
    for node in range(1, n):
        if parent[node] >= 0 and np.isfinite(d[node, parent[node]]):
            tree[node, parent[node]] = True
            tree[parent[node], node] = True

    tree_degree = tree.sum(axis=1).astype(float)

    # Compute the maximum tree edge on every tree path.
    path_max = np.zeros((n, n), dtype=float)
    component = np.full(n, -1, dtype=int)
    component_id = 0

    for start in range(n):
        if component[start] >= 0:
            continue

        component[start] = component_id
        stack = [(start, -1, 0.0)]

        while stack:
            node, previous, current_max = stack.pop()
            path_max[start, node] = current_max

            neighbors = np.where(tree[node])[0]
            for neighbor in neighbors:
                if neighbor == previous:
                    continue
                if component[neighbor] < 0:
                    component[neighbor] = component_id
                    next_max = max(current_max, d[node, neighbor])
                    stack.append((neighbor, node, next_max))

        component_id += 1

    # Alpha-nearness: small excess over the strongest edge on the tree path
    # indicates that the edge is a plausible alternative in a tour.
    alpha = np.zeros((n, n), dtype=float)
    same_component = component[:, None] == component[None, :]
    alpha[same_component] = np.maximum(
        d[same_component] - path_max[same_component],
        0.0,
    )
    alpha[~same_component] = d[~same_component]

    finite_alpha = alpha[np.isfinite(alpha) & (alpha > 0)]
    alpha_scale = (
        float(np.median(finite_alpha))
        if finite_alpha.size
        else distance_scale
    )
    alpha_scale = max(alpha_scale, np.finfo(float).eps)

    # Convert alpha values into a smooth priority without using rank signals.
    alpha_score = 1.0 / (1.0 + alpha / alpha_scale)

    # Mildly discourage edges attached to highly branched MST vertices.
    branch_pressure = 1.0 / (
        1.0
        + 0.35
        * np.maximum(tree_degree - 2.0, 0.0)
    )
    degree_factor = np.sqrt(
        branch_pressure[:, None] * branch_pressure[None, :]
    )

    # Preserve a weak direct-cost component for disconnected or degenerate cases.
    direct_score = 1.0 / (1.0 + d / distance_scale)

    score = (
        0.78 * alpha_score
        + 0.14 * degree_factor * alpha_score
        + 0.08 * direct_score
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
