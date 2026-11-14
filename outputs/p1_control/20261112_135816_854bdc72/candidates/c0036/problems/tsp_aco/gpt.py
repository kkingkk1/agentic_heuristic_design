import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    distances = 0.5 * (distances + distances.T)
    distances = np.maximum(distances, 0.0)

    valid = np.isfinite(distances)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid & (distances > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    edge_indices = np.argwhere(np.triu(valid, k=1))
    edge_indices = sorted(
        edge_indices.tolist(),
        key=lambda pair: (distances[pair[0], pair[1]], pair[0], pair[1]),
    )

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)
    tree_edges = np.zeros((n, n), dtype=bool)
    forest = [[] for _ in range(n)]

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

    for i, j in edge_indices:
        if union(i, j):
            weight = float(distances[i, j])
            tree_edges[i, j] = True
            tree_edges[j, i] = True
            forest[i].append((j, weight))
            forest[j].append((i, weight))

    bottleneck = np.full((n, n), np.inf, dtype=float)

    for source in range(n):
        stack = [(source, -np.inf, -1)]
        while stack:
            node, current_max, previous = stack.pop()
            bottleneck[source, node] = (
                0.0 if node == source else current_max
            )

            for neighbor, weight in forest[node]:
                if neighbor != previous:
                    stack.append(
                        (neighbor, max(current_max, weight), node)
                    )

    score = np.zeros((n, n), dtype=float)

    for i, j in edge_indices:
        distance = float(distances[i, j])

        if tree_edges[i, j]:
            value = 1.0
        else:
            path_bottleneck = bottleneck[i, j]
            if np.isfinite(path_bottleneck):
                excess = max(0.0, distance - path_bottleneck)
                value = float(np.exp(-excess / scale))
            else:
                value = 0.0

        score[i, j] = value
        score[j, i] = value

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
