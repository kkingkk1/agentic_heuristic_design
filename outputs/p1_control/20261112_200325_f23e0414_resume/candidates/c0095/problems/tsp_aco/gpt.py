import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-replacement and topology-aware edge-priority indicators."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[finite_positive]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    work = np.asarray(distances, dtype=float).copy()
    work[~np.isfinite(work)] = fallback
    work[work <= 0.0] = fallback
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work) & (work > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_work = work[valid]
    scale = float(np.median(finite_work))
    scale = max(scale, np.finfo(float).eps)

    parent = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf, dtype=float)
    used = np.zeros(n, dtype=bool)
    best[0] = 0.0

    tree = np.zeros((n, n), dtype=bool)

    for _ in range(n):
        candidates = np.where(~used, best, np.inf)
        node = int(np.argmin(candidates))

        if not np.isfinite(candidates[node]):
            break

        used[node] = True
        predecessor = parent[node]
        if predecessor >= 0:
            tree[node, predecessor] = True
            tree[predecessor, node] = True

        neighbors = (~used) & valid[node]
        improved = neighbors & (work[node] < best)
        best[improved] = work[node, improved]
        parent[improved] = node

    tree_degree = tree.sum(axis=1).astype(float)

    path_max = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(path_max, 0.0)

    for source in range(n):
        stack = [(source, -1, 0.0)]
        while stack:
            node, predecessor, current_max = stack.pop()
            path_max[source, node] = current_max

            neighbors = np.flatnonzero(tree[node])
            for neighbor in neighbors:
                if neighbor == predecessor:
                    continue
                edge_max = max(current_max, work[node, neighbor])
                stack.append((int(neighbor), node, edge_max))

    excess = np.maximum(work - path_max, 0.0)
    excess[~np.isfinite(excess)] = fallback
    excess[~valid] = 0.0

    relative_excess = excess / (scale + np.finfo(float).eps)
    replacement_score = np.exp(-np.clip(relative_excess, 0.0, 50.0))

    degree_pressure = np.abs(tree_degree - 2.0)
    endpoint_balance = 1.0 / (
        1.0
        + 0.35
        * (degree_pressure[:, None] + degree_pressure[None, :])
    )

    tree_bonus = np.where(tree, 1.18, 1.0)

    score = replacement_score * endpoint_balance * tree_bonus
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
