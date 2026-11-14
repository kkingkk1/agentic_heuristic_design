import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a global rooted-1-tree consensus prior for TSP edge sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative values")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    d = distances.copy()
    np.fill_diagonal(d, np.inf)

    support = np.zeros((n, n), dtype=float)

    def add_edge(i: int, j: int) -> None:
        if i != j:
            support[i, j] += 1.0
            support[j, i] += 1.0

    def rooted_one_tree(root: int) -> None:
        active = np.array([i for i in range(n) if i != root], dtype=int)
        m = active.size

        if m == 0:
            return

        in_tree = np.zeros(m, dtype=bool)
        best_cost = np.full(m, np.inf, dtype=float)
        parent = np.full(m, -1, dtype=int)

        best_cost[0] = 0.0

        for _ in range(m):
            candidates = np.where(~in_tree, best_cost, np.inf)
            pos = int(np.argmin(candidates))
            if not np.isfinite(candidates[pos]):
                break

            in_tree[pos] = True
            node = int(active[pos])

            if parent[pos] >= 0:
                add_edge(node, int(parent[pos]))

            remaining = np.flatnonzero(~in_tree)
            if remaining.size:
                remaining_nodes = active[remaining]
                improved = d[node, remaining_nodes] < best_cost[remaining]
                changed = remaining[improved]

                best_cost[changed] = d[node, active[changed]]
                parent[changed] = node

        root_edges = np.argsort(d[root, active], kind="stable")[: min(2, m)]
        for pos in root_edges:
            add_edge(root, int(active[pos]))

    for root in range(n):
        rooted_one_tree(root)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    distance_tiebreak = np.exp(-d / scale)
    distance_tiebreak[~np.isfinite(distance_tiebreak)] = 0.0
    np.fill_diagonal(distance_tiebreak, 0.0)

    support /= float(max(n, 1))
    score = support + 1e-6 * distance_tiebreak
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
