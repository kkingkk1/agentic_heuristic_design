import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-exchange-based edge priorities for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(finite_positive, False)

    positive = distances[finite_positive]
    fallback = float(np.max(positive)) if positive.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    off_diagonal = d[~np.eye(n, dtype=bool)]
    positive_off_diagonal = off_diagonal[off_diagonal > 0.0]
    scale = (
        float(np.median(positive_off_diagonal))
        if positive_off_diagonal.size
        else fallback
    )
    scale = max(scale, np.finfo(float).eps)

    if n == 2:
        result = np.zeros((2, 2), dtype=float)
        result[0, 1] = result[1, 0] = 1.0
        return result

    # Deterministic Prim minimum spanning tree.
    parent = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf, dtype=float)
    chosen = np.zeros(n, dtype=bool)
    best[0] = 0.0

    for _ in range(n):
        candidate = np.where(chosen, np.inf, best)
        u = int(np.argmin(candidate))
        chosen[u] = True

        for v in range(n):
            if not chosen[v] and d[u, v] < best[v]:
                best[v] = d[u, v]
                parent[v] = u

    tree = [[] for _ in range(n)]
    tree_degree = np.zeros(n, dtype=float)

    for v in range(1, n):
        u = int(parent[v])
        if u < 0:
            u = 0
        weight = float(d[u, v])
        tree[u].append((v, weight))
        tree[v].append((u, weight))
        tree_degree[u] += 1.0
        tree_degree[v] += 1.0

    # Largest tree-edge weight on every unique tree path.
    path_max = np.zeros((n, n), dtype=float)

    for source in range(n):
        visited = np.zeros(n, dtype=bool)
        visited[source] = True
        stack = [(source, 0.0)]

        while stack:
            node, current_max = stack.pop()
            path_max[source, node] = current_max

            for neighbor, weight in tree[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(
                        (neighbor, max(current_max, weight))
                    )

    # Small nonnegative slack indicates a cheap tree exchange.
    exchange_slack = np.maximum(d - path_max, 0.0)

    # Mildly favor shorter edges after accounting for exchange structure.
    length_factor = np.exp(-0.30 * np.minimum(d / scale, 40.0))
    exchange_factor = np.exp(
        -np.minimum(exchange_slack / scale, 40.0)
    )

    # Penalize endpoints whose spanning-tree degree already exceeds TSP degree two.
    excess_degree = np.maximum(tree_degree - 2.0, 0.0)
    degree_factor = np.exp(
        -0.12 * (excess_degree[:, None] + excess_degree[None, :])
    )

    score = exchange_factor * length_factor * degree_factor
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
