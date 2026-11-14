import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n < 2:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    symmetric_d = 0.5 * (d + d.T)
    valid &= np.isfinite(symmetric_d) & (symmetric_d >= 0.0)

    positive = symmetric_d[valid & (symmetric_d > 0.0)]
    distance_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    distance_scale = max(distance_scale, np.finfo(float).eps)

    clean = np.where(valid, symmetric_d, np.inf)
    np.fill_diagonal(clean, np.inf)

    parent = np.full(n, -1, dtype=int)
    selected = np.zeros(n, dtype=bool)
    component = np.full(n, -1, dtype=int)
    tree = np.zeros((n, n), dtype=bool)

    component_id = 0

    for start in range(n):
        if selected[start]:
            continue

        best = np.full(n, np.inf, dtype=float)
        best[start] = 0.0

        while True:
            candidates = np.where(~selected, best, np.inf)
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            selected[node] = True
            component[node] = component_id

            if parent[node] >= 0:
                p = parent[node]
                tree[node, p] = True
                tree[p, node] = True

            neighbors = (~selected) & np.isfinite(clean[node])
            improved = neighbors & (clean[node] < best)
            parent[improved] = node
            best[improved] = clean[node, improved]

        component_id += 1

    degree = tree.sum(axis=1).astype(float)
    degree_scale = max(1.0, float(np.sqrt(n)))
    degree_pressure = np.exp(
        -(degree[:, None] + degree[None, :]) / degree_scale
    )

    bottleneck = np.full((n, n), np.inf, dtype=float)

    for source in range(n):
        if component[source] < 0:
            continue

        stack = [source]
        visited = np.zeros(n, dtype=bool)
        visited[source] = True
        bottleneck[source, source] = 0.0

        while stack:
            node = stack.pop()
            children = np.where(tree[node] & ~visited)[0]

            for child in children:
                visited[child] = True
                bottleneck[source, child] = max(
                    bottleneck[source, node],
                    clean[node, child],
                )
                stack.append(int(child))

    score = np.zeros((n, n), dtype=float)
    same_component = (
        (component[:, None] >= 0)
        & (component[:, None] == component[None, :])
    )

    tree_bonus = tree.astype(float)

    replaceable = (
        valid
        & same_component
        & ~tree
        & np.isfinite(bottleneck)
    )

    slack = np.maximum(
        clean - bottleneck,
        0.0,
    )

    exchange_score = np.exp(
        -np.minimum(slack / distance_scale, 60.0)
    )

    score[replaceable] = exchange_score[replaceable]
    score += 0.8 * tree_bonus
    score *= degree_pressure
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
