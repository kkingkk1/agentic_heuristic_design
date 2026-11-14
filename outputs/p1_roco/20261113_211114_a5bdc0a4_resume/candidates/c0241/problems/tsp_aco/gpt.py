import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-stability-based edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid_directed = finite & nonnegative
    valid = valid_directed & valid_directed.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.zeros((n, n), dtype=float)
    symmetric[valid] = 0.5 * (raw[valid] + raw.T[valid])

    positive = symmetric[valid & (symmetric > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    finite_values = symmetric[valid]
    largest = float(np.max(finite_values)) if finite_values.size else scale
    penalty = largest + 20.0 * scale

    clean = np.full((n, n), penalty, dtype=float)
    clean[valid] = symmetric[valid]
    np.fill_diagonal(clean, penalty)

    edges_i, edges_j = np.triu_indices(n, k=1)
    edge_valid = valid[edges_i, edges_j]

    edges_i = edges_i[edge_valid]
    edges_j = edges_j[edge_valid]
    edge_distances = clean[edges_i, edges_j]

    edge_count = edges_i.size
    if edge_count == 0:
        return np.zeros((n, n), dtype=float)

    base_cost = edge_distances / max(scale, eps)

    pair_code = (
        (edges_i.astype(np.int64) + 1) * 73856093
        ^ (edges_j.astype(np.int64) + 1) * 19349663
        ^ (n + 1) * 83492791
    ).astype(np.float64)

    trials = int(np.clip(12 + 2.5 * np.sqrt(float(n)), 16, 40))
    counts = np.zeros(edge_count, dtype=float)

    def find(parent: np.ndarray, node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]

        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node

        return root

    for trial in range(trials):
        phase = (
            pair_code * 0.000001
            + (trial + 1) * 1.61803398875
            + 0.37 * np.cos(pair_code * 0.00000017)
        )
        perturbation = 1.0 + 0.20 * np.sin(phase)
        costs = base_cost * perturbation

        order = np.lexsort((edges_j, edges_i, costs))
        parent = np.arange(n, dtype=int)
        rank = np.zeros(n, dtype=np.int8)

        selected = 0
        for edge_index in order:
            left = int(edges_i[edge_index])
            right = int(edges_j[edge_index])

            root_left = find(parent, left)
            root_right = find(parent, right)

            if root_left == root_right:
                continue

            if rank[root_left] < rank[root_right]:
                parent[root_left] = root_right
            elif rank[root_left] > rank[root_right]:
                parent[root_right] = root_left
            else:
                parent[root_right] = root_left
                rank[root_left] += 1

            counts[edge_index] += 1.0
            selected += 1

            if selected == n - 1:
                break

    stability = counts / float(max(trials, 1))

    affinity = np.exp(
        -np.minimum(edge_distances / max(scale, eps), 50.0)
    )
    affinity /= max(float(np.max(affinity)), eps)

    incident_stability = np.zeros(n, dtype=float)
    for index in range(edge_count):
        incident_stability[edges_i[index]] += stability[index]
        incident_stability[edges_j[index]] += stability[index]

    incident_scale = np.maximum(
        incident_stability,
        np.finfo(float).eps,
    )

    endpoint_balance = np.sqrt(
        (
            stability
            + 0.10 / incident_scale[edges_i]
        )
        * (
            stability
            + 0.10 / incident_scale[edges_j]
        )
    )

    score_edges = (
        0.64 * stability
        + 0.26 * affinity
        + 0.10 * endpoint_balance
    )

    result = np.zeros((n, n), dtype=float)
    result[edges_i, edges_j] = score_edges
    result[edges_j, edges_i] = score_edges
    np.fill_diagonal(result, 0.0)

    result[~valid] = 0.0
    result = np.maximum(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
