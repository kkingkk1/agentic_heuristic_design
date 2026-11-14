import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric MST-exchange edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    off_diagonal = ~np.eye(n, dtype=bool)
    positive = distances[finite & off_diagonal & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[off_diagonal & np.isfinite(d)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(finite_offdiag))
    scale = max(scale, np.finfo(float).eps)

    parent = np.full(n, -1, dtype=int)
    key = np.full(n, np.inf, dtype=float)
    in_tree = np.zeros(n, dtype=bool)
    key[0] = 0.0

    for _ in range(n):
        candidates = np.where(~in_tree, key, np.inf)
        u = int(np.argmin(candidates))
        if not np.isfinite(candidates[u]):
            break

        in_tree[u] = True
        neighbors = np.where(~in_tree & (d[u] < key))[0]
        key[neighbors] = d[u, neighbors]
        parent[neighbors] = u

    tree = [[] for _ in range(n)]
    tree_edge = np.zeros((n, n), dtype=bool)

    for v in range(1, n):
        u = parent[v]
        if u >= 0:
            tree[u].append((v, d[u, v]))
            tree[v].append((u, d[u, v]))
            tree_edge[u, v] = True
            tree_edge[v, u] = True

    path_max = np.zeros((n, n), dtype=float)

    for source in range(n):
        stack = [(source, -1, 0.0)]
        while stack:
            node, previous, current_max = stack.pop()
            path_max[source, node] = current_max

            for neighbor, weight in tree[node]:
                if neighbor != previous:
                    stack.append(
                        (neighbor, node, max(current_max, weight))
                    )

    finite_path = np.maximum(path_max, np.finfo(float).eps)
    exchange_slack = np.maximum(d - finite_path, 0.0)

    tree_strength = tree_edge.astype(float)
    replacement_strength = np.exp(
        -np.clip(exchange_slack / scale, 0.0, 50.0)
    )

    distance_strength = np.exp(
        -np.clip((d / scale) ** 0.75, 0.0, 50.0)
    )

    degree = tree_edge.sum(axis=1).astype(float)
    target_degree = 2.0
    degree_pressure = 1.0 + 0.35 * np.maximum(
        target_degree - degree[:, None],
        0.0,
    )
    degree_pressure *= 1.0 + 0.35 * np.maximum(
        target_degree - degree[None, :],
        0.0,
    )

    score = (
        (0.35 + 0.65 * replacement_strength)
        * (0.55 + 0.45 * distance_strength)
        * degree_pressure
    )
    score *= 0.65 + 0.35 * tree_strength

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
