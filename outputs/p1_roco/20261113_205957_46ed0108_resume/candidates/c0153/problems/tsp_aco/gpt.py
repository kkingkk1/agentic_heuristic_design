import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural edge priors based on minimum 1-tree support."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    diagonal = np.eye(n, dtype=bool)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(valid, 0.5 * (raw + raw.T), np.inf)
    np.fill_diagonal(cleaned, np.inf)

    positive = cleaned[np.isfinite(cleaned) & (cleaned > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    support = np.zeros((n, n), dtype=float)
    quality_sum = 0.0

    for root in range(n):
        active = np.ones(n, dtype=bool)
        active[root] = False

        in_tree = np.zeros(n, dtype=bool)
        in_tree[root] = True
        key = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)

        first = int(np.flatnonzero(active)[0])
        key[first] = 0.0

        tree_cost = 0.0
        tree_edges = []

        for _ in range(n - 1):
            candidates = np.where(active & ~in_tree, key, np.inf)
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            in_tree[node] = True

            if parent[node] >= 0:
                u, v = parent[node], node
                a, b = min(u, v), max(u, v)
                tree_edges.append((a, b))
                tree_cost += float(cleaned[u, v])

            neighbors = np.where(active & ~in_tree, cleaned[node], np.inf)
            improve = neighbors < key
            key[improve] = neighbors[improve]
            parent[improve] = node

        root_candidates = cleaned[root].copy()
        root_candidates[~active] = np.inf
        root_order = np.argsort(root_candidates, kind="stable")

        chosen_root_edges = []
        for node in root_order:
            if np.isfinite(root_candidates[node]):
                a, b = min(root, int(node)), max(root, int(node))
                chosen_root_edges.append((a, b))
                tree_cost += float(root_candidates[node])
                if len(chosen_root_edges) == min(2, n - 1):
                    break

        one_tree_edges = tree_edges + chosen_root_edges
        if not one_tree_edges:
            continue

        quality = 1.0 / (1.0 + tree_cost / scale)
        quality_sum += quality

        for a, b in one_tree_edges:
            support[a, b] += quality
            support[b, a] += quality

    if quality_sum > 0.0 and np.isfinite(quality_sum):
        support /= quality_sum

    support[~valid] = 0.0
    np.fill_diagonal(support, 0.0)

    maximum = float(np.max(support))
    if np.isfinite(maximum) and maximum > 0.0:
        support /= maximum

    return np.nan_to_num(
        support,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
