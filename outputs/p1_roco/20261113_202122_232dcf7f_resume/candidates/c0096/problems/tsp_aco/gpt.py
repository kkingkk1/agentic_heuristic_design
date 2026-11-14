import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structure-aware edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & finite.T & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.maximum(distances, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)
    cleaned[~valid] = np.inf
    np.fill_diagonal(cleaned, np.inf)

    positive = cleaned[valid & (cleaned > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    forest = np.zeros((n, n), dtype=bool)
    component = np.full(n, -1, dtype=int)
    component_id = 0

    for start in range(n):
        if component[start] >= 0:
            continue

        stack = [start]
        component[start] = component_id
        members = []

        while stack:
            node = stack.pop()
            members.append(node)
            neighbors = np.flatnonzero(valid[node] & (component < 0))
            for neighbor in neighbors:
                component[neighbor] = component_id
                stack.append(int(neighbor))

        member_mask = np.zeros(n, dtype=bool)
        member_mask[members] = True
        in_tree = np.zeros(n, dtype=bool)
        best = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best[start] = 0.0

        for _ in range(len(members)):
            candidates = np.where(member_mask & ~in_tree, best, np.inf)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                break

            in_tree[node] = True
            if parent[node] >= 0:
                p = parent[node]
                forest[node, p] = True
                forest[p, node] = True

            neighbors = np.flatnonzero(
                member_mask & ~in_tree & valid[node]
            )
            improved = cleaned[node, neighbors] < best[neighbors]
            selected = neighbors[improved]
            best[selected] = cleaned[node, selected]
            parent[selected] = node

        component_id += 1

    tree_edges = np.argwhere(np.triu(forest, 1))
    tree_edge_set = set((int(i), int(j)) for i, j in tree_edges)

    bottleneck = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(bottleneck, 0.0)

    for root in range(n):
        seen = np.zeros(n, dtype=bool)
        seen[root] = True
        stack = [root]

        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(forest[node] & ~seen)
            for neighbor in neighbors:
                seen[neighbor] = True
                bottleneck[root, neighbor] = max(
                    bottleneck[root, node],
                    cleaned[node, neighbor],
                )
                stack.append(int(neighbor))

    criticality = np.zeros((n, n), dtype=float)

    for u, v in tree_edges:
        u = int(u)
        v = int(v)

        seen = np.zeros(n, dtype=bool)
        seen[u] = True
        stack = [u]

        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(forest[node] & ~seen)
            for neighbor in neighbors:
                if (node == u and neighbor == v) or (
                    node == v and neighbor == u
                ):
                    continue
                seen[neighbor] = True
                stack.append(int(neighbor))

        crossing = valid & (seen[:, None] != seen[None, :])
        alternatives = cleaned[crossing]
        if alternatives.size:
            replacement = float(np.min(alternatives))
            gap = max(replacement - cleaned[u, v], 0.0)
            value = 1.0 - np.exp(-gap / scale)
        else:
            value = 1.0

        criticality[u, v] = value
        criticality[v, u] = value

    cycle_gain = np.zeros((n, n), dtype=float)
    same_component = component[:, None] == component[None, :]
    usable_path = valid & same_component & np.isfinite(bottleneck)

    gain = np.maximum(bottleneck - cleaned, 0.0)
    cycle_gain[usable_path] = 1.0 - np.exp(
        -gain[usable_path] / scale
    )

    centered_distance = np.zeros((n, n), dtype=float)
    centered_distance[valid] = np.exp(
        -cleaned[valid] / (scale + eps)
    )

    tree_component = criticality
    non_tree_component = cycle_gain

    score = np.where(
        forest,
        0.58 * tree_component + 0.22 * centered_distance,
        0.58 * non_tree_component + 0.22 * centered_distance,
    )

    score += 0.20 * np.sqrt(
        np.maximum(tree_component * tree_component.T, 0.0)
        + np.maximum(non_tree_component * non_tree_component.T, 0.0)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
