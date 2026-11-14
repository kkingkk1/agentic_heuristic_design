import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global MST-cycle-exchange edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n < 2:
        return result

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    mutual_valid = finite_nonnegative & finite_nonnegative.T
    if not np.any(mutual_valid):
        return result

    symmetric = 0.5 * (distances + distances.T)
    valid_values = symmetric[mutual_valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    edges = [
        (float(symmetric[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if mutual_valid[i, j]
    ]
    edges.sort(key=lambda item: item[0])

    forest = [[] for _ in range(n)]
    forest_edges = set()

    for weight, i, j in edges:
        if union(i, j):
            forest[i].append((j, weight))
            forest[j].append((i, weight))
            forest_edges.add((i, j))

    for i in range(n):
        for j in range(i + 1, n):
            if not mutual_valid[i, j]:
                continue

            path_max = None
            stack = [(i, -np.inf, -1)]
            visited = {i}

            while stack:
                node, current_max, previous = stack.pop()

                if node == j:
                    path_max = current_max
                    break

                for neighbor, weight in forest[node]:
                    if neighbor != previous and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(
                            (neighbor, max(current_max, weight), node)
                        )

            edge_weight = float(symmetric[i, j])

            if path_max is None:
                excess = edge_weight / scale
            else:
                excess = max(0.0, (edge_weight - path_max) / scale)

            priority = float(np.exp(-min(excess, 60.0)))

            if (i, j) in forest_edges:
                priority = 1.0

            result[i, j] = priority
            result[j, i] = priority

    np.fill_diagonal(result, 0.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
