import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d)
    valid = finite & finite.T & ~np.eye(n, dtype=bool)
    valid &= d >= 0.0
    valid &= d.T >= 0.0

    if not np.any(valid):
        return result

    cost = 0.5 * (np.where(valid, d, 0.0) + np.where(valid, d.T, 0.0))
    cost[~valid] = np.inf

    positive = cost[valid & (cost > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    support = np.zeros((n, n), dtype=float)
    fallback = np.exp(-np.where(valid, cost, np.inf) / scale)
    fallback[~valid] = 0.0

    for excluded in range(n):
        active = np.ones(n, dtype=bool)
        active[excluded] = False

        tree = [[] for _ in range(n)]
        used = np.zeros(n, dtype=bool)

        for start in np.flatnonzero(active):
            if used[start]:
                continue

            used[start] = True
            best = np.full(n, np.inf, dtype=float)
            parent = np.full(n, -1, dtype=int)
            best[start] = 0.0

            while True:
                candidates = np.where(active & ~used, best, np.inf)
                node = int(np.argmin(candidates))
                if not np.isfinite(candidates[node]):
                    break

                used[node] = True
                p = parent[node]
                if p >= 0:
                    tree[node].append(p)
                    tree[p].append(node)

                distances = cost[node]
                improved = active & ~used & (distances < best)
                best[improved] = distances[improved]
                parent[improved] = node

        for source in np.flatnonzero(active):
            stack = [(int(source), -np.inf, -1)]
            seen = {int(source)}

            while stack:
                node, path_max, previous = stack.pop()

                if node != source and valid[source, node]:
                    if cost[source, node] <= path_max:
                        value = 1.0
                    else:
                        gap = cost[source, node] - path_max
                        value = float(np.exp(-gap / scale))
                    support[source, node] += value

                for neighbor in tree[node]:
                    if neighbor == previous or neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_max = max(path_max, cost[node, neighbor])
                    stack.append((neighbor, next_max, node))

    support /= max(n, 1)

    incident_count = np.maximum(
        valid.sum(axis=0) + valid.sum(axis=1),
        1,
    )
    structural = support / np.sqrt(
        incident_count[:, None] * incident_count[None, :]
    )
    structural = 0.5 * (structural + structural.T)

    result = 0.8 * structural + 0.2 * fallback
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result)) if result.size else 0.0
    if maximum > 0.0:
        result /= maximum

    return result.astype(float, copy=False)
