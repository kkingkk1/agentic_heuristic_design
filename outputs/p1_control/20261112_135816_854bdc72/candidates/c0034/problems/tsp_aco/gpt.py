import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    off_diagonal = ~np.eye(n, dtype=bool)
    usable = finite & off_diagonal

    positive = d[usable & (d > 0)]
    if positive.size:
        fallback = float(np.median(positive))
    else:
        finite_values = d[usable & finite]
        fallback = float(np.median(finite_values)) if finite_values.size else 1.0

    fallback = max(fallback, np.finfo(float).eps)

    clean = np.where(usable, d, fallback * 10.0)
    clean = np.maximum(clean, 0.0)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, np.inf)

    finite_edges = clean[np.isfinite(clean) & off_diagonal]
    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(finite_edges))
    scale = max(scale, np.finfo(float).eps)

    # Prim minimum spanning tree.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    tree = np.zeros((n, n), dtype=bool)

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        u = int(np.argmin(candidates))
        if not np.isfinite(candidates[u]):
            break

        in_tree[u] = True
        p = parent[u]
        if p >= 0:
            tree[u, p] = True
            tree[p, u] = True

        improve = (~in_tree) & (clean[u] < best)
        best[improve] = clean[u, improve]
        parent[improve] = u

    tree_weight = np.where(tree, clean, 0.0)
    tree_degree = np.sum(tree, axis=1).astype(float)
    degree_target = 2.0
    degree_balance = 1.0 / (
        1.0
        + 0.35
        * np.abs(tree_degree[:, None] - degree_target)
        * np.abs(tree_degree[None, :] - degree_target)
    )

    # Maximum edge weight on the unique tree path for every endpoint pair.
    bottleneck = np.zeros((n, n), dtype=float)
    for start in range(n):
        stack = [(start, -1, 0.0)]
        while stack:
            node, previous, current_max = stack.pop()
            bottleneck[start, node] = current_max

            neighbors = np.flatnonzero(tree[node])
            for nxt in neighbors:
                if nxt != previous:
                    stack.append(
                        (
                            int(nxt),
                            node,
                            max(current_max, float(clean[node, nxt])),
                        )
                    )

    # Non-tree edges are favored when they create low-slack alternative cycles.
    cycle_slack = np.maximum(clean - bottleneck, 0.0)
    cycle_signal = np.exp(-cycle_slack / scale)
    cycle_signal[tree] = 1.0

    # For every tree edge, estimate the cheapest replacement crossing its cut.
    replacement = np.full((n, n), np.inf, dtype=float)

    for a in range(n):
        for b in range(a + 1, n):
            if not tree[a, b]:
                continue

            blocked = tree.copy()
            blocked[a, b] = False
            blocked[b, a] = False

            component = np.zeros(n, dtype=bool)
            stack = [a]
            component[a] = True

            while stack:
                node = stack.pop()
                for nxt in np.flatnonzero(blocked[node]):
                    nxt = int(nxt)
                    if not component[nxt]:
                        component[nxt] = True
                        stack.append(nxt)

            crossing = component[:, None] ^ component[None, :]
            crossing &= off_diagonal
            crossing &= np.isfinite(clean)

            alternatives = clean[crossing]
            if alternatives.size:
                replacement[a, b] = replacement[b, a] = float(
                    np.min(alternatives)
                )

    tree_edges = tree & np.isfinite(replacement)
    replacement_margin = np.zeros((n, n), dtype=float)
    replacement_margin[tree_edges] = np.maximum(
        replacement[tree_edges] - clean[tree_edges],
        0.0,
    ) / scale

    criticality = np.zeros((n, n), dtype=float)
    criticality[tree_edges] = 1.0 - np.exp(-replacement_margin[tree_edges])

    # Third-node replacement difficulty.
    positive_floor = max(np.finfo(float).eps, scale * 1e-12)
    detour_gap = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isfinite(clean[i, j]):
                continue

            intermediates = np.array(
                [
                    clean[i, k] + clean[k, j]
                    for k in range(n)
                    if k != i and k != j
                    and np.isfinite(clean[i, k])
                    and np.isfinite(clean[k, j])
                ],
                dtype=float,
            )

            if intermediates.size:
                best_detour = float(np.min(intermediates))
                gap = max(best_detour - clean[i, j], 0.0)
                value = 1.0 - np.exp(-gap / (clean[i, j] + positive_floor))
            else:
                value = 0.0

            detour_gap[i, j] = value
            detour_gap[j, i] = value

    score = (
        0.42 * cycle_signal
        + 0.30 * criticality
        + 0.20 * detour_gap
        + 0.08 * degree_balance
    )

    score[~off_diagonal] = 0.0
    score[~np.isfinite(score)] = 0.0
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
