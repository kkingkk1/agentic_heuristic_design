import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(raw) & (raw > 0.0)
    symmetric = np.where(
        np.isfinite(raw) & np.isfinite(raw.T),
        0.5 * (raw + raw.T),
        np.inf,
    )
    valid = np.isfinite(symmetric) & (symmetric > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = symmetric[valid]
    fallback = max(float(np.max(positive_values)), eps)
    replacement = max(10.0 * fallback, eps)

    work = np.where(valid, symmetric, replacement)
    np.fill_diagonal(work, replacement)

    # Two cheapest incident edges provide a local TSP connection baseline.
    ordered = np.sort(work, axis=1)
    first = ordered[:, 0]
    second = ordered[:, min(1, n - 2)]

    endpoint_savings = (
        (first[:, None] + first[None, :])
        + (second[:, None] + second[None, :])
        - 2.0 * work
    )
    savings_scale = max(float(np.max(first + second)), eps)
    savings_score = np.clip(
        0.5 + 0.5 * endpoint_savings / savings_scale,
        0.0,
        1.0,
    )

    # Build a deterministic minimum spanning tree with Prim's algorithm.
    tree = np.zeros((n, n), dtype=bool)
    attached = np.zeros(n, dtype=bool)
    best = np.full(n, replacement, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    for _ in range(n):
        masked_best = np.where(attached, np.inf, best)
        node = int(np.argmin(masked_best))
        if not np.isfinite(masked_best[node]):
            break

        attached[node] = True
        predecessor = parent[node]
        if predecessor >= 0:
            tree[node, predecessor] = True
            tree[predecessor, node] = True

        candidate = work[node]
        improve = (~attached) & (candidate < best)
        best[improve] = candidate[improve]
        parent[improve] = node

    tree_score = tree.astype(float)

    # Compute shortest-path distances on the tree using repeated relaxation.
    tree_distance = np.full((n, n), replacement, dtype=float)
    np.fill_diagonal(tree_distance, 0.0)

    for source in range(n):
        distance = np.full(n, replacement, dtype=float)
        distance[source] = 0.0
        visited = np.zeros(n, dtype=bool)

        for _ in range(n):
            choices = np.where(visited, np.inf, distance)
            current = int(np.argmin(choices))
            if visited[current] or not np.isfinite(choices[current]):
                break

            visited[current] = True
            neighbors = np.flatnonzero(tree[current])
            if neighbors.size:
                trial = distance[current] + work[current, neighbors]
                update = trial < distance[neighbors]
                distance[neighbors[update]] = trial[update]

        tree_distance[source] = distance

    # Prefer edges that are economical shortcuts over their tree route.
    detour_ratio = work / np.maximum(tree_distance, eps)
    shortcut_score = 1.0 / (1.0 + np.maximum(detour_ratio - 1.0, 0.0))
    shortcut_score = np.clip(shortcut_score, 0.0, 1.0)

    # Triangle closure rewards edges supported by short two-hop alternatives.
    two_hop = np.min(work[:, :, None] + work[None, :, :], axis=1)
    triangle_ratio = work / np.maximum(two_hop, eps)
    triangle_score = 1.0 / (1.0 + np.maximum(triangle_ratio - 1.0, 0.0))
    triangle_score = np.clip(triangle_score, 0.0, 1.0)

    score = (
        0.48 * tree_score
        + 0.24 * savings_score
        + 0.18 * shortcut_score
        + 0.10 * triangle_score
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
