import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-cycle-aware edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    raw = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    raw = np.maximum(raw, 0.0)
    raw = 0.5 * (raw + raw.T)

    finite_offdiag = raw[~diagonal & np.isfinite(raw)]
    positive = finite_offdiag[finite_offdiag > 0.0]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(raw), raw, fallback)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = fallback
    scale = max(scale, eps)

    # Prim minimum spanning tree.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    tree = [[] for _ in range(n)]

    for _ in range(n):
        masked_best = np.where(in_tree, np.inf, best)
        node = int(np.argmin(masked_best))

        if not np.isfinite(masked_best[node]):
            remaining = np.flatnonzero(~in_tree)
            node = int(remaining[0])

        in_tree[node] = True
        p = int(parent[node])

        if p >= 0:
            weight = float(d[node, p])
            tree[node].append((p, weight))
            tree[p].append((node, weight))

        improved = (~in_tree) & (d[node] < best)
        parent[improved] = node
        best[improved] = d[node, improved]

    # Maximum edge weight on each tree path.
    path_max = np.zeros((n, n), dtype=float)

    for source in range(n):
        stack = [(source, -1, 0.0)]

        while stack:
            node, previous, current_max = stack.pop()
            path_max[source, node] = current_max

            for neighbor, weight in tree[node]:
                if neighbor != previous:
                    stack.append(
                        (
                            neighbor,
                            node,
                            max(current_max, float(weight)),
                        )
                    )

    tree_bottleneck = np.maximum(path_max, eps)

    # Cycle excess: how much worse an edge is than the heaviest
    # tree edge it would replace.
    excess = np.maximum(d - tree_bottleneck, 0.0)
    excess_ratio = excess / scale

    distance_ratio = d / scale

    global_attraction = np.exp(
        -np.clip(distance_ratio ** 0.7, 0.0, 40.0)
    )

    cycle_quality = np.exp(
        -np.clip(1.8 * excess_ratio ** 0.9, 0.0, 40.0)
    )

    # Penalize edges incident to nodes already surrounded by many
    # globally attractive alternatives, reducing hub concentration.
    base_affinity = np.exp(
        -np.clip(distance_ratio ** 0.8, 0.0, 40.0)
    )
    np.fill_diagonal(base_affinity, 0.0)

    node_load = np.sum(base_affinity, axis=1)
    load_scale = float(np.median(node_load[node_load > 0.0]))
    load_scale = max(load_scale, eps)

    endpoint_balance = 1.0 / np.sqrt(
        (1.0 + node_load[:, None] / load_scale)
        * (1.0 + node_load[None, :] / load_scale)
    )

    score = (
        global_attraction
        * cycle_quality
        * (0.55 + 0.45 * endpoint_balance)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
