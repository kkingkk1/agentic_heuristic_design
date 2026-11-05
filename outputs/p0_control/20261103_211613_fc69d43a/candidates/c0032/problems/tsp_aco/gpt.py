import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally structure-aware edge scores for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    d = 0.5 * (distances + distances.T)
    np.fill_diagonal(d, np.inf)

    usable = d[np.isfinite(d)]
    if usable.size == 0:
        return scores

    positive = usable[usable > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Prim's algorithm for a deterministic minimum spanning tree.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        node = int(np.argmin(candidates))
        in_tree[node] = True

        if parent[node] >= 0:
            # The complete finite graph guarantees a valid parent.
            pass

        updated = d[node]
        improve = (~in_tree) & (updated < best)
        best[improve] = updated[improve]
        parent[improve] = node

    tree = [[] for _ in range(n)]
    for node in range(1, n):
        ancestor = parent[node]
        if ancestor < 0:
            continue
        weight = float(d[node, ancestor])
        tree[node].append((ancestor, weight))
        tree[ancestor].append((node, weight))

    # max_on_path[i, j] is the largest tree edge on the unique tree path.
    max_on_path = np.zeros((n, n), dtype=float)

    for source in range(n):
        stack = [(source, -1, 0.0)]
        while stack:
            node, previous, path_max = stack.pop()
            max_on_path[source, node] = path_max
            for neighbor, weight in tree[node]:
                if neighbor != previous:
                    stack.append(
                        (neighbor, node, max(path_max, weight))
                    )

    # Edges with large positive slack are replaceable by a shorter tree route.
    tree_bottleneck = max_on_path
    redundancy = np.maximum(d - tree_bottleneck, 0.0)

    # Prefer short edges, while strongly discounting structurally redundant ones.
    raw = np.exp(-d / scale) * np.exp(-redundancy / scale)
    raw[~np.isfinite(raw)] = 0.0
    np.fill_diagonal(raw, 0.0)

    maximum = float(np.max(raw))
    if maximum > 0.0:
        raw /= maximum

    return raw.astype(float, copy=False)
