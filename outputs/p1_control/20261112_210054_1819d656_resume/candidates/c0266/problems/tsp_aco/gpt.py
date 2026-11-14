import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-exchange-based edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[np.isfinite(d)]
    replacement = (
        float(np.max(finite_offdiag))
        if finite_offdiag.size
        else fallback
    )
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = fallback

    d = np.where(np.isfinite(d), d, replacement)
    np.fill_diagonal(d, np.inf)

    scale = float(np.median(d[np.isfinite(d)]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(fallback, np.finfo(float).eps)

    # Prim's algorithm for a deterministic minimum spanning tree.
    parent = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf, dtype=float)
    used = np.zeros(n, dtype=bool)
    best[0] = 0.0

    for _ in range(n):
        available = np.where(~used, best, np.inf)
        node = int(np.argmin(available))
        if used[node] or not np.isfinite(best[node]):
            break

        used[node] = True
        if node != 0:
            weight = d[node, parent[node]]
            edge_weight = float(weight)
        else:
            edge_weight = 0.0

        candidates = d[node]
        improve = (~used) & (candidates < best)
        parent[improve] = node
        best[improve] = candidates[improve]

    tree = np.zeros((n, n), dtype=bool)
    for node in range(1, n):
        p = parent[node]
        if p >= 0:
            tree[node, p] = True
            tree[p, node] = True

    tree_weight = np.zeros((n, n), dtype=float)
    tree_weight[tree] = d[tree]

    # Compute the maximum edge weight on every MST path.
    path_max = np.zeros((n, n), dtype=float)

    adjacency = [[] for _ in range(n)]
    for node in range(1, n):
        p = parent[node]
        if p >= 0:
            w = float(d[node, p])
            adjacency[node].append((p, w))
            adjacency[p].append((node, w))

    for source in range(n):
        stack = [(source, -1, 0.0)]
        while stack:
            current, previous, current_max = stack.pop()
            path_max[source, current] = current_max

            for neighbor, weight in adjacency[current]:
                if neighbor != previous:
                    stack.append(
                        (
                            neighbor,
                            current,
                            max(current_max, weight),
                        )
                    )

    eps = np.finfo(float).eps
    path_scale = np.maximum(path_max, eps)

    # Exchange quality: a non-tree edge is useful when it competes with
    # the bottleneck edge on its MST path.
    stretch = d / path_scale
    exchange_quality = 1.0 / (1.0 + np.maximum(stretch - 1.0, 0.0))

    # Edges shorter than their tree-path bottleneck receive a modest bonus.
    improvement = np.maximum(path_max - d, 0.0)
    improvement_ratio = improvement / (path_scale + d + eps)
    improvement_bonus = 1.0 + 0.35 * np.clip(improvement_ratio, 0.0, 1.0)

    # Keep tree edges structurally important without making the prior sparse.
    structural_support = np.where(tree, 0.82, 0.0)
    structural_support += np.where(~tree, 0.18 * exchange_quality, 0.0)

    # A weak global length term breaks ties without using endpoint-local scales.
    global_length = np.exp(
        -np.clip(np.maximum(d, 0.0) / (3.0 * scale + eps), 0.0, 20.0)
    )
    global_length = 0.75 + 0.25 * global_length

    # Penalize edges incident to unusually high-degree MST hubs slightly.
    degrees = tree.sum(axis=1).astype(float)
    degree_factor = 1.0 / np.sqrt(
        (1.0 + degrees[:, None]) * (1.0 + degrees[None, :])
    )
    degree_factor = 0.88 + 0.12 * degree_factor / np.maximum(
        np.max(degree_factor),
        eps,
    )

    score = (
        structural_support
        * improvement_bonus
        * global_length
        * degree_factor
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
