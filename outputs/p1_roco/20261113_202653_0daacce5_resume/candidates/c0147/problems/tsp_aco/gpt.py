import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural-connectivity priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    directed_valid = finite & nonnegative
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid | directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)

    both_valid = directed_valid & directed_valid.T
    distances[both_valid] = (
        raw[both_valid] + raw.T[both_valid]
    ) * 0.5

    only_forward = directed_valid & ~directed_valid.T
    distances[only_forward] = raw[only_forward]

    only_backward = directed_valid.T & ~directed_valid
    distances[only_backward] = raw.T[only_backward]

    np.fill_diagonal(distances, np.inf)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    edge_list = [
        (float(distances[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if valid[i, j] and np.isfinite(distances[i, j])
    ]

    if not edge_list:
        return np.zeros((n, n), dtype=float)

    edge_list.sort(key=lambda item: (item[0], item[1], item[2]))

    parent = np.arange(n, dtype=int)
    component_size = np.ones(n, dtype=float)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]

        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node

        return root

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)

        if root_first == root_second:
            return

        if component_size[root_first] < component_size[root_second]:
            root_first, root_second = root_second, root_first

        parent[root_second] = root_first
        component_size[root_first] += component_size[root_second]

    connectivity = np.zeros((n, n), dtype=float)

    index = 0
    while index < len(edge_list):
        next_index = index + 1
        current_distance = edge_list[index][0]

        while (
            next_index < len(edge_list)
            and edge_list[next_index][0] == current_distance
        ):
            next_index += 1

        pending_unions = []

        for _, i, j in edge_list[index:next_index]:
            root_i = find(i)
            root_j = find(j)

            if root_i != root_j:
                gain = (
                    component_size[root_i]
                    * component_size[root_j]
                    / float(n * n)
                )
                connectivity[i, j] = gain
                connectivity[j, i] = gain
                pending_unions.append((i, j))

        for i, j in pending_unions:
            union(i, j)

        index = next_index

    finite_edges = distances[valid]
    positive_edges = finite_edges[finite_edges > 0.0]
    kernel_scale = (
        float(np.quantile(positive_edges, 0.60))
        if positive_edges.size
        else scale
    )
    kernel_scale = max(kernel_scale, 0.10 * scale, np.finfo(float).eps)

    proximity = np.zeros((n, n), dtype=float)
    proximity[valid] = np.exp(
        -distances[valid] / kernel_scale
    )

    structural = connectivity * (0.55 + 0.45 * proximity)

    fallback = 0.025 * proximity
    score = np.where(connectivity > 0.0, structural, fallback)
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
