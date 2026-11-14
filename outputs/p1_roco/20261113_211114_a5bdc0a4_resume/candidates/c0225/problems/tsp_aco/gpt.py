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
    positive_values = edge_values[edge_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    scale = max(scale, np.finfo(float).eps)

    ordered = np.sort(edge_values)
    ranks = np.searchsorted(ordered, edge_values, side="right")
    percentile = ranks.astype(float) / float(edge_values.size)

    global_preference = np.zeros((n, n), dtype=float)
    global_preference[valid] = 0.12 + 0.88 * (
        1.0 - percentile
    )

    median_distance = float(np.median(edge_values))
    dispersion = float(
        np.median(np.abs(edge_values - median_distance))
    )
    temperature = max(
        dispersion,
        scale * 0.08,
        np.finfo(float).eps,
    )

    centered = (median_distance - symmetric) / temperature
    logistic = np.zeros((n, n), dtype=float)
    logistic[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -40.0, 40.0))
    )
    logistic[valid] = 0.18 + 0.82 * logistic[valid]

    # Vectorized best two-hop replacement computation.
    # The intermediate node is processed in vectorized batches,
    # avoiding the original Python loop over every edge and node.
    best_two_hop = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        through_k = (
            symmetric[:, k, None] + symmetric[None, k, :]
        )
        best_two_hop = np.minimum(best_two_hop, through_k)

    direct = np.maximum(symmetric, scale * 1e-12)
    advantage = best_two_hop / direct
    efficiency = np.tanh(
        np.maximum(advantage - 1.0, 0.0)
    )
    efficiency[~np.isfinite(efficiency)] = 1.0
    efficiency[~valid] = 0.0

    cycle_support = np.zeros((n, n), dtype=float)
    cycle_support[valid] = 0.24 + 0.76 * efficiency[valid]

    # Prim minimum spanning forest.
    tree_edges = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)
    best_cost = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for component_start in range(n):
        if visited[component_start]:
            continue

        best_cost[component_start] = 0.0

        while True:
            candidates = np.where(
                visited,
                np.inf,
                best_cost,
            )
            u = int(np.argmin(candidates))

            if not np.isfinite(candidates[u]):
                break

            visited[u] = True

            if parent[u] >= 0:
                v = parent[u]
                tree_edges[u, v] = True
                tree_edges[v, u] = True

            neighbors = np.flatnonzero(
                valid[u] & ~visited
            )
            improved = symmetric[u, neighbors] < best_cost[neighbors]
            update_nodes = neighbors[improved]

            best_cost[update_nodes] = symmetric[
                u, update_nodes
            ]
            parent[update_nodes] = u

    # Mutual nearest-neighbor edges provide a local consistency signal
    # without relying on endpoint density as the primary structure.
    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size:
            nearest[i] = neighbors[
                np.argmin(symmetric[i, neighbors])
            ]

    mutual_nearest = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = True
            mutual_nearest[j, i] = True

    connectivity = np.full((n, n), 0.62, dtype=float)
    connectivity[tree_edges] = 1.24
    connectivity[mutual_nearest] *= 1.10
    connectivity[~valid] = 0.0

    eps = np.finfo(float).eps
    score = (
        np.maximum(global_preference, eps)
        * np.maximum(logistic, eps)
        * np.maximum(cycle_support, eps)
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
