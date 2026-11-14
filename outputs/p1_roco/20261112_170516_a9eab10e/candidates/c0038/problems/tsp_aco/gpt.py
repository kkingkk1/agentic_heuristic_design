import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global MST/alpha-nearness edge desirability indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    symmetric = 0.5 * (distances + distances.T)
    valid = np.isfinite(symmetric) & (symmetric > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive = symmetric[valid]
    fallback = float(np.median(positive))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    safe = np.where(valid, symmetric, np.inf)
    safe = np.where(np.isfinite(safe), safe, fallback * 100.0)
    np.fill_diagonal(safe, 0.0)

    tree = np.zeros((n, n), dtype=bool)
    parent = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf, dtype=float)
    best[0] = 0.0
    chosen = np.zeros(n, dtype=bool)

    for _ in range(n):
        candidates = np.where(~chosen, best, np.inf)
        node = int(np.argmin(candidates))

        if chosen[node] or not np.isfinite(candidates[node]):
            break

        chosen[node] = True

        if parent[node] >= 0:
            p = parent[node]
            tree[node, p] = True
            tree[p, node] = True

        neighbors = np.where(~chosen)[0]
        improved = safe[node, neighbors] < best[neighbors]
        update_nodes = neighbors[improved]
        best[update_nodes] = safe[node, update_nodes]
        parent[update_nodes] = node

    adjacency = [[] for _ in range(n)]
    tree_edges = safe[tree]
    for i, j in zip(*np.where(np.triu(tree, 1))):
        weight = float(safe[i, j])
        adjacency[i].append((j, weight))
        adjacency[j].append((i, weight))

    path_max = np.zeros((n, n), dtype=float)

    for source in range(n):
        stack = [(source, -1, 0.0)]
        while stack:
            node, previous, current_max = stack.pop()
            path_max[source, node] = current_max

            for neighbor, weight in adjacency[node]:
                if neighbor != previous:
                    stack.append(
                        (neighbor, node, max(current_max, weight))
                    )

    scale_values = positive[np.isfinite(positive)]
    scale = float(np.median(scale_values)) if scale_values.size else fallback
    scale = max(scale, np.finfo(float).eps)

    alpha = safe + path_max
    alpha[tree] = safe[tree] * 0.5
    alpha[~valid] = np.inf
    np.fill_diagonal(alpha, np.inf)

    score = np.exp(-np.clip(alpha / scale, 0.0, 80.0))
    score[tree] *= 1.35
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))

    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
