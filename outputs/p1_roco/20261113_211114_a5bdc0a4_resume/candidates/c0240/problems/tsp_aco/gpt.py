import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    directed_valid = finite & nonnegative
    directed_valid &= ~np.eye(n, dtype=bool)

    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (
        distances[valid] + distances.T[valid]
    )
    np.fill_diagonal(symmetric, np.inf)

    edge_values = symmetric[valid]
    positive = edge_values[edge_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    ordered = np.sort(edge_values)
    ranks = np.searchsorted(ordered, edge_values, side="right")
    percentile = ranks.astype(float) / max(float(edge_values.size), 1.0)

    global_preference = np.zeros((n, n), dtype=float)
    global_preference[valid] = 1.0 - percentile
    global_preference[valid] = 0.20 + 0.80 * global_preference[valid]

    median_distance = float(np.median(edge_values))
    dispersion = float(np.median(np.abs(edge_values - median_distance)))
    temperature = max(
        dispersion * 0.85,
        scale * 0.06,
        np.finfo(float).eps,
    )

    centered = (median_distance - symmetric) / temperature
    logistic = np.zeros((n, n), dtype=float)
    logistic[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -40.0, 40.0))
    )
    logistic[valid] = 0.12 + 0.88 * logistic[valid]

    detour = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not valid[i, j]:
                continue

            best_two_hop = np.inf
            for k in range(n):
                if k == i or k == j:
                    continue
                if valid[i, k] and valid[k, j]:
                    candidate = symmetric[i, k] + symmetric[k, j]
                    if candidate < best_two_hop:
                        best_two_hop = candidate

            if np.isfinite(best_two_hop):
                direct = max(symmetric[i, j], scale * 1e-12)
                advantage = best_two_hop / direct
                efficiency = np.tanh(
                    max(advantage - 1.0, 0.0) * 1.15
                )
            else:
                efficiency = 1.0

            detour[i, j] = efficiency
            detour[j, i] = efficiency

    cycle_support = np.zeros((n, n), dtype=float)
    cycle_support[valid] = 0.18 + 0.82 * detour[valid]

    tree_edges = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)
    best_cost = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for component_start in range(n):
        if visited[component_start]:
            continue

        best_cost[component_start] = 0.0

        while True:
            candidates = np.where(~visited, best_cost, np.inf)
            u = int(np.argmin(candidates))

            if not np.isfinite(candidates[u]):
                break

            visited[u] = True

            if parent[u] >= 0:
                v = parent[u]
                tree_edges[u, v] = True
                tree_edges[v, u] = True

            neighbors = np.where(valid[u] & ~visited)[0]
            for v in neighbors:
                if symmetric[u, v] < best_cost[v]:
                    best_cost[v] = symmetric[u, v]
                    parent[v] = u

    connectivity = np.full((n, n), 0.72, dtype=float)
    connectivity[tree_edges] = 1.18
    connectivity[~valid] = 0.0

    eps = np.finfo(float).eps
    score = (
        np.maximum(global_preference, eps) ** 0.90
        * np.maximum(logistic, eps) ** 1.15
        * np.maximum(cycle_support, eps) ** 1.10
        * connectivity
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
