import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global MST-exchange priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = np.maximum(np.where(finite, distances, 0.0), 0.0)

    valid = finite & finite.T
    valid &= ~np.eye(n, dtype=bool)

    d = np.full((n, n), np.inf, dtype=float)
    d[valid] = 0.5 * (
        nonnegative[valid] + nonnegative.T[valid]
    )
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    forest = [[] for _ in range(n)]
    component = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    tree_edge = np.zeros((n, n), dtype=bool)

    component_id = 0

    for root in range(n):
        if visited[root]:
            continue

        visited[root] = True
        component[root] = component_id
        best = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best[root] = 0.0

        while True:
            candidates = np.where(~visited, best, np.inf)
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            visited[node] = True
            component[node] = component_id

            p = int(parent[node])
            if p >= 0:
                weight = float(d[node, p])
                forest[node].append((p, weight))
                forest[p].append((node, weight))
                tree_edge[node, p] = True
                tree_edge[p, node] = True

            neighbors = np.where(~visited & np.isfinite(d[node]))[0]
            improved = d[node, neighbors] < best[neighbors]
            selected = neighbors[improved]

            best[selected] = d[node, selected]
            parent[selected] = node

        component_id += 1

    bottleneck = np.full((n, n), np.inf, dtype=float)

    for source in range(n):
        bottleneck[source, source] = 0.0
        stack = [(source, -1, 0.0)]

        while stack:
            node, parent, current_max = stack.pop()

            for neighbor, weight in forest[node]:
                if neighbor == parent:
                    continue

                next_max = max(current_max, weight)
                bottleneck[source, neighbor] = next_max
                stack.append((neighbor, node, next_max))

    same_component = (
        component[:, None] == component[None, :]
    ) & valid

    exchange_delta = np.zeros((n, n), dtype=float)
    usable = same_component & np.isfinite(bottleneck)
    exchange_delta[usable] = bottleneck[usable] - d[usable]

    score = np.zeros((n, n), dtype=float)
    score[usable] = np.exp(
        np.clip(exchange_delta[usable] / scale, -50.0, 50.0)
    )

    score[tree_edge] = 1.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
