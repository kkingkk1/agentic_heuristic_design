import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a global tree-and-tour stability prior for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[finite_positive]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    directed = np.where(finite_positive, distances, np.inf)
    clean = np.minimum(directed, directed.T)

    allowed = np.isfinite(clean) & (clean > 0)
    np.fill_diagonal(allowed, False)

    clean = np.where(allowed, clean, 4.0 * fallback)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, np.inf)

    tree_count = np.zeros((n, n), dtype=float)
    tour_count = np.zeros((n, n), dtype=float)

    variants = max(6, min(14, n + 4))

    for variant in range(variants):
        indices = np.arange(n, dtype=float)
        left = indices[:, None] + 1.0
        right = indices[None, :] + 1.0

        phase = (
            0.73 * left * right
            + 1.17 * (left + right) * (variant + 1)
            + 0.41 * (left - right) ** 2
        )
        perturbation = 1.0 + 0.18 * np.sin(phase)
        perturbation = 0.5 * (perturbation + perturbation.T)
        np.fill_diagonal(perturbation, 1.0)

        cost = clean * perturbation
        np.fill_diagonal(cost, np.inf)

        root = variant % n
        in_tree = np.zeros(n, dtype=bool)
        best_cost = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best_cost[root] = 0.0

        for _ in range(n):
            candidates = np.where(~in_tree, best_cost, np.inf)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                break

            in_tree[node] = True
            unseen = ~in_tree
            improved = unseen & (cost[node] < best_cost)
            parent[improved] = node
            best_cost[improved] = cost[node, improved]

        adjacency = [[] for _ in range(n)]

        for node in range(n):
            predecessor = int(parent[node])
            if predecessor < 0 or predecessor == node:
                continue

            if allowed[node, predecessor]:
                tree_count[node, predecessor] += 1.0
                tree_count[predecessor, node] += 1.0

            adjacency[node].append(predecessor)
            adjacency[predecessor].append(node)

        for neighbors in adjacency:
            neighbors.sort()

        preorder = []
        visited = np.zeros(n, dtype=bool)
        stack = [root]

        while stack:
            node = stack.pop()
            if visited[node]:
                continue

            visited[node] = True
            preorder.append(node)

            for neighbor in reversed(adjacency[node]):
                if not visited[neighbor]:
                    stack.append(neighbor)

        for node in range(n):
            if not visited[node]:
                preorder.append(node)

        if len(preorder) >= 2:
            for position, source in enumerate(preorder):
                target = preorder[(position + 1) % len(preorder)]
                if source == target or not allowed[source, target]:
                    continue

                tour_count[source, target] += 1.0
                tour_count[target, source] += 1.0

    tree_score = tree_count / float(max(variants, 1))
    tour_score = tour_count / float(max(variants, 1))

    score = 0.35 * tree_score + 0.65 * tour_score
    score = 0.5 * (score + score.T)
    score[~allowed] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(score))

    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
