import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a topology-aware normalized edge-priority matrix for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    directed_valid = finite_nonnegative & ~diagonal

    usable = directed_valid | directed_valid.T
    if not np.any(usable):
        return np.zeros((n, n), dtype=float)

    inf = np.inf
    edge_distance = np.full((n, n), inf, dtype=float)

    forward = directed_valid
    backward = directed_valid.T

    edge_distance[forward] = distances[forward]
    edge_distance[backward] = np.minimum(
        edge_distance[backward],
        distances.T[backward],
    )
    edge_distance[~usable] = inf
    np.fill_diagonal(edge_distance, inf)

    finite_values = edge_distance[usable]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0
    scale = max(scale, eps)

    # Build a deterministic minimum spanning forest with Kruskal's algorithm.
    upper_edges = [
        (float(edge_distance[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if np.isfinite(edge_distance[i, j])
    ]
    upper_edges.sort(key=lambda item: (item[0], item[1], item[2]))

    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    forest = np.zeros((n, n), dtype=bool)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False

        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a

        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1
        return True

    adjacency = [[] for _ in range(n)]

    for weight, i, j in upper_edges:
        if union(i, j):
            forest[i, j] = True
            forest[j, i] = True
            adjacency[i].append((j, weight))
            adjacency[j].append((i, weight))

    # Compute the maximum edge weight on each forest path.
    path_bottleneck = np.full((n, n), inf, dtype=float)

    for source in range(n):
        path_bottleneck[source, source] = 0.0
        stack = [(source, -1, 0.0)]

        while stack:
            node, predecessor, current_max = stack.pop()
            path_bottleneck[source, node] = current_max

            for neighbor, weight in adjacency[node]:
                if neighbor != predecessor:
                    stack.append(
                        (neighbor, node, max(current_max, weight))
                    )

    scores = np.zeros((n, n), dtype=float)

    for i, j in zip(*np.where(np.triu(usable, 1))):
        weight = edge_distance[i, j]
        bottleneck = path_bottleneck[i, j]

        if forest[i, j]:
            # Forest edges provide direct global connectivity.
            score = np.exp(-weight / scale)
        elif np.isfinite(bottleneck):
            # A non-tree edge is useful when it is a cheap cycle exchange.
            excess = max(weight - bottleneck, 0.0)
            replacement_quality = np.exp(-excess / scale)
            stretch_penalty = 1.0 / (
                1.0 + max(weight - bottleneck, 0.0) / scale
            )
            score = replacement_quality * stretch_penalty
        else:
            # Edges between disconnected components are not usable in
            # the current minimum spanning forest topology.
            score = 0.0

        scores[i, j] = score
        scores[j, i] = score

    scores[~usable] = 0.0
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
