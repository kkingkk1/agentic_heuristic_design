import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge heuristic based on spanning-tree exchange potential."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    d = 0.5 * (raw + raw.T)
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    usable = d[valid]
    if usable.size == 0:
        return result

    scale = float(np.median(usable))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = usable[usable > 0.0]
        scale = float(np.median(positive)) if positive.size else 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    work = np.where(valid, d, np.inf)
    np.fill_diagonal(work, np.inf)

    parent = np.full(n, -1, dtype=int)
    component = np.full(n, -1, dtype=int)
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf, dtype=float)
    forest_edges = []

    component_id = 0

    for start in range(n):
        if in_tree[start]:
            continue

        key[start] = 0.0

        while True:
            candidates = np.where(~in_tree & np.isfinite(key))[0]
            if candidates.size == 0:
                break

            u = int(candidates[np.argmin(key[candidates])])
            in_tree[u] = True
            component[u] = component_id

            if parent[u] >= 0:
                v = parent[u]
                forest_edges.append((v, u))
                result[v, u] = 1.0
                result[u, v] = 1.0

            neighbors = np.where(~in_tree & np.isfinite(work[u]))[0]
            for v in neighbors:
                if work[u, v] < key[v]:
                    key[v] = work[u, v]
                    parent[v] = u

        component_id += 1

    if not forest_edges:
        return result

    for u, v in forest_edges:
        result[u, v] = 1.0
        result[v, u] = 1.0

    bottleneck = np.full((n, n), np.inf, dtype=float)

    adjacency = [[] for _ in range(n)]
    for u, v in forest_edges:
        weight = work[u, v]
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    for start in range(n):
        if component[start] < 0:
            continue

        stack = [(start, -1, 0.0)]
        while stack:
            u, previous, path_max = stack.pop()
            bottleneck[start, u] = path_max

            for v, weight in adjacency[u]:
                if v != previous:
                    stack.append((v, u, max(path_max, weight)))

    non_tree = valid & (result == 0.0)

    if np.any(non_tree):
        edge_values = d[non_tree]
        path_values = bottleneck[non_tree]

        connected_non_tree = np.isfinite(path_values)
        if np.any(connected_non_tree):
            excess = np.maximum(
                edge_values[connected_non_tree]
                - path_values[connected_non_tree],
                0.0,
            )

            exchange_scores = np.exp(-excess / (scale + eps))
            rows, cols = np.where(non_tree)
            selected_rows = rows[connected_non_tree]
            selected_cols = cols[connected_non_tree]

            result[selected_rows, selected_cols] = exchange_scores
            result[selected_cols, selected_rows] = exchange_scores

    np.fill_diagonal(result, 0.0)
    result[~np.isfinite(result)] = 0.0
    result[~valid] = 0.0

    values = result[valid]
    if values.size:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum > minimum:
            result[valid] = (values - minimum) / (maximum - minimum)
        else:
            result[valid] = 1.0

    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)
    result[~np.isfinite(result)] = 0.0

    return result.astype(float, copy=False)
