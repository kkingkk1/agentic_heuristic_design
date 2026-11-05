import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric MST-backbone edge-score matrix for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    if np.any(distances < 0.0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    work = 0.5 * (distances + distances.T)
    np.fill_diagonal(work, np.inf)

    finite_edges = work[np.isfinite(work)]
    if finite_edges.size == 0:
        return scores

    positive_edges = finite_edges[finite_edges > 0.0]
    scale = (
        float(np.median(positive_edges))
        if positive_edges.size
        else 1.0
    )
    eps = np.finfo(float).eps * max(scale, 1.0)

    finite_max = float(np.max(finite_edges))
    replacement = max(finite_max, scale, 1.0)
    work = np.where(np.isfinite(work), work, replacement)
    np.fill_diagonal(work, np.inf)

    parent = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf, dtype=float)
    used = np.zeros(n, dtype=bool)
    best[0] = 0.0

    tree = [[] for _ in range(n)]

    for _ in range(n):
        candidate = np.where(used, np.inf, best)
        current = int(np.argmin(candidate))
        used[current] = True

        if parent[current] >= 0:
            previous = int(parent[current])
            weight = float(work[current, previous])
            tree[current].append((previous, weight))
            tree[previous].append((current, weight))

        updates = (~used) & (work[current] < best)
        parent[updates] = current
        best[updates] = work[current, updates]

    bottleneck = np.zeros((n, n), dtype=float)

    for source in range(n):
        stack = [(source, -1, 0.0)]
        while stack:
            node, previous, current_max = stack.pop()
            bottleneck[source, node] = current_max

            for neighbor, edge_weight in tree[node]:
                if neighbor != previous:
                    next_max = max(current_max, edge_weight)
                    stack.append((neighbor, node, next_max))

    direct = np.where(np.isfinite(work), work, replacement)
    redundancy = np.maximum(direct - bottleneck, 0.0)

    backbone_term = np.exp(-redundancy / max(scale, eps))
    distance_term = 1.0 / (1.0 + direct / max(scale, eps))

    scores = backbone_term * distance_term
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
