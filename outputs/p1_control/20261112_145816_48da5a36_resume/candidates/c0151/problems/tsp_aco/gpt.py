import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    valid = finite & finite.T & (d > 0.0) & (d.T > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    work = np.where(valid, symmetric_d, np.inf)

    positive = work[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)), eps)

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
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
        (float(work[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if valid[i, j]
    ]
    edges.sort(key=lambda item: item[0])

    forest = [[] for _ in range(n)]
    forest_degree = np.zeros(n, dtype=float)

    for weight, i, j in edges:
        if union(i, j):
            forest[i].append((j, weight))
            forest[j].append((i, weight))
            forest_degree[i] += 1.0
            forest_degree[j] += 1.0

    components = np.array([find(i) for i in range(n)])
    bottleneck = np.zeros((n, n), dtype=float)

    for root in range(n):
        stack = [(root, -1, 0.0)]
        while stack:
            node, previous, path_max = stack.pop()
            bottleneck[root, node] = path_max
            for neighbor, weight in forest[node]:
                if neighbor != previous:
                    stack.append(
                        (neighbor, node, max(path_max, weight))
                    )

    same_component = components[:, None] == components[None, :]
    connected_path = same_component & (bottleneck > 0.0)

    slack = np.zeros((n, n), dtype=float)
    slack[connected_path] = np.maximum(
        work[connected_path] - bottleneck[connected_path],
        0.0,
    )

    positive_slack = slack[valid & (slack > 0.0)]
    slack_scale = (
        float(np.median(positive_slack))
        if positive_slack.size
        else global_scale
    )
    slack_scale = max(slack_scale, eps)

    replacement_score = np.exp(-slack / slack_scale)
    length_score = 1.0 / (1.0 + work / global_scale)

    degree_balance = 1.0 / (
        1.0
        + 0.25
        * (forest_degree[:, None] + forest_degree[None, :])
    )

    component_factor = np.where(
        same_component,
        1.0,
        0.2 * np.exp(-work / global_scale),
    )

    score = (
        replacement_score
        * length_score
        * degree_balance
        * component_factor
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
