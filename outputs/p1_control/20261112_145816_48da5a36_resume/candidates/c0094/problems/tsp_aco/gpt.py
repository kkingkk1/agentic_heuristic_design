import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric TSP edge-priority matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = 0.5 * (d + d.T)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid & (d > 0.0)]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, eps)

    safe = np.where(valid, d, np.inf)

    row_order = np.argsort(safe, axis=1, kind="stable")
    sorted_values = np.take_along_axis(safe, row_order, axis=1)

    best = sorted_values[:, 0]
    second = sorted_values[:, min(1, n - 1)]

    best = np.where(np.isfinite(best), best, scale)
    second = np.where(np.isfinite(second), second, best + scale)

    endpoint_scale = 0.5 * (
        best[:, None] + best[None, :] + 2.0 * scale
    )

    edge_distance = np.where(valid, d, scale * 10.0)

    regret_i = np.maximum(second[:, None] - edge_distance, 0.0)
    regret_j = np.maximum(second[None, :] - edge_distance, 0.0)

    regret = 0.5 * (
        regret_i / (second[:, None] + scale)
        + regret_j / (second[None, :] + scale)
    )
    regret = np.clip(regret, 0.0, 1.0)

    relative_length = edge_distance / (
        endpoint_scale + eps
    )

    scores = np.exp(
        2.4 * regret - 0.75 * np.minimum(relative_length, 50.0)
    )

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)

        if root_left == root_right:
            return False

        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left

        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    upper = np.triu(valid, k=1)
    edge_rows, edge_cols = np.nonzero(upper)

    if edge_rows.size:
        edge_values = d[edge_rows, edge_cols]
        order = np.argsort(edge_values, kind="stable")

        backbone = np.zeros((n, n), dtype=bool)
        degree = np.zeros(n, dtype=float)

        for position in order:
            i = int(edge_rows[position])
            j = int(edge_cols[position])

            if union(i, j):
                backbone[i, j] = True
                backbone[j, i] = True
                degree[i] += 1.0
                degree[j] += 1.0

        degree_factor = 1.0 / (
            1.0
            + 0.35 * (
                np.maximum(degree[:, None] - 1.0, 0.0)
                + np.maximum(degree[None, :] - 1.0, 0.0)
            )
        )

        scores *= degree_factor
        scores *= np.where(backbone, 2.75, 1.0)

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    scores[~np.isfinite(scores)] = 0.0
    return scores.astype(float, copy=False)
