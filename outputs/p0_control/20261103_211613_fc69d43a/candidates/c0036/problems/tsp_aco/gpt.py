import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-cycle-criticality edge scores for a minimization TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if np.any(np.isfinite(distances) & (distances < 0)):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    valid = finite & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    d = np.where(finite, distances, np.inf)
    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & ~np.eye(n, dtype=bool)

    positive = d[valid & (d > 0.0)]
    distance_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    distance_scale = max(distance_scale, np.finfo(float).eps)

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

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

    edge_rows, edge_cols = np.where(np.triu(valid, k=1))
    edge_order = np.argsort(d[edge_rows, edge_cols], kind="stable")

    tree = [[] for _ in range(n)]
    for index in edge_order:
        i = int(edge_rows[index])
        j = int(edge_cols[index])

        if union(i, j):
            weight = float(d[i, j])
            tree[i].append((j, weight))
            tree[j].append((i, weight))

    bottleneck = np.full((n, n), np.inf, dtype=float)

    for source in range(n):
        stack = [(source, -np.inf, -1)]

        while stack:
            node, path_max, previous = stack.pop()
            bottleneck[source, node] = path_max

            for neighbor, weight in tree[node]:
                if neighbor != previous:
                    stack.append(
                        (neighbor, max(path_max, weight), node)
                    )

    cycle_slack = np.zeros_like(d, dtype=float)
    connected = np.isfinite(bottleneck) & valid
    cycle_slack[connected] = np.maximum(
        d[connected] - bottleneck[connected],
        0.0,
    )

    slack_values = cycle_slack[connected]
    slack_scale = (
        float(np.median(slack_values[slack_values > 0.0]))
        if np.any(slack_values > 0.0)
        else distance_scale
    )
    slack_scale = max(slack_scale, np.finfo(float).eps)

    absolute_term = np.exp(-d / distance_scale)
    criticality_term = np.exp(-cycle_slack / slack_scale)

    scores = absolute_term * criticality_term
    scores[~connected] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
