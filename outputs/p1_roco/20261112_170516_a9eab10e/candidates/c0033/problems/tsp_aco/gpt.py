import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST alpha-nearness edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    symmetric_valid = finite_positive & finite_positive.T

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    d = np.zeros((n, n), dtype=float)
    d[symmetric_valid] = 0.5 * (
        distances[symmetric_valid] + distances.T[symmetric_valid]
    )
    np.fill_diagonal(symmetric_valid, False)
    np.fill_diagonal(d, np.inf)

    edges = [
        (float(d[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if symmetric_valid[i, j]
    ]
    edges.sort(key=lambda item: (item[0], item[1], item[2]))

    parent = np.arange(n)
    size = np.ones(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    tree = [[] for _ in range(n)]
    for weight, i, j in edges:
        if union(i, j):
            tree[i].append((j, weight))
            tree[j].append((i, weight))

    component = np.array([find(i) for i in range(n)])

    path_max = np.full((n, n), np.inf, dtype=float)

    for source in range(n):
        stack = [(source, -np.inf, -1)]
        while stack:
            node, current_max, previous = stack.pop()
            path_max[source, node] = current_max
            for neighbor, weight in tree[node]:
                if neighbor != previous:
                    stack.append(
                        (neighbor, max(current_max, weight), node)
                    )

    positive_values = d[symmetric_valid]
    scale = float(np.median(positive_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    alpha = d - path_max
    same_component = component[:, None] == component[None, :]
    usable = symmetric_valid & same_component & np.isfinite(alpha)

    alpha = np.maximum(alpha, 0.0)
    score = np.zeros((n, n), dtype=float)

    structural_scale = max(scale, np.finfo(float).eps)
    score[usable] = np.exp(-alpha[usable] / structural_scale)

    tree_edge = np.isfinite(path_max) & symmetric_valid & (alpha <= 1e-12)
    score[tree_edge] = 1.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
