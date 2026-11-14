import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Estimate TSP edge priorities from perturbed minimum-spanning-tree samples."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    symmetric_input = np.allclose(
        distances,
        distances.T,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    pair_valid = valid & valid.T
    if symmetric_input:
        pair_valid = valid

    undirected_valid = pair_valid | pair_valid.T
    if not np.any(undirected_valid):
        return result

    edge_cost = np.full((n, n), np.inf, dtype=float)

    both = pair_valid & pair_valid.T
    edge_cost[both] = 0.5 * (
        distances[both] + distances.T[both]
    )

    only_forward = pair_valid & ~both
    edge_cost[only_forward] = distances[only_forward]

    only_backward = pair_valid.T & ~both
    edge_cost[only_backward] = distances.T[only_backward]

    if not symmetric_input:
        edge_cost[undirected_valid] = np.where(
            np.isfinite(edge_cost[undirected_valid]),
            edge_cost[undirected_valid],
            edge_cost.T[undirected_valid],
        )

    finite_costs = edge_cost[undirected_valid]
    order = np.argsort(finite_costs, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(order.size, dtype=float)

    if order.size > 1:
        rank_values = ranks / float(order.size - 1)
    else:
        rank_values = np.zeros(1, dtype=float)

    rank_matrix = np.full((n, n), np.inf, dtype=float)
    rank_matrix[undirected_valid] = rank_values

    edge_i, edge_j = np.nonzero(
        np.triu(undirected_valid, k=1)
    )

    if edge_i.size == 0:
        edge_i, edge_j = np.nonzero(undirected_valid)
        keep = edge_i != edge_j
        edge_i, edge_j = edge_i[keep], edge_j[keep]

    edge_costs = edge_cost[edge_i, edge_j]
    edge_ranks = rank_matrix[edge_i, edge_j]

    sample_count = int(np.clip(8 + 2 * np.log2(max(n, 2)), 8, 24))
    rng = np.random.default_rng(1729)

    frequency = np.zeros(edge_i.size, dtype=float)
    degree_sum = np.zeros(n, dtype=float)

    def find(parent: np.ndarray, node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    for _ in range(sample_count):
        noise = rng.gumbel(
            loc=0.0,
            scale=0.18 + 0.22 * np.maximum(edge_ranks, 0.0),
            size=edge_i.size,
        )
        perturbed = edge_ranks + noise
        sequence = np.argsort(perturbed, kind="stable")

        parent = np.arange(n, dtype=int)
        degree = np.zeros(n, dtype=float)
        chosen = np.zeros(edge_i.size, dtype=bool)
        chosen_count = 0

        for position in sequence:
            left = int(edge_i[position])
            right = int(edge_j[position])
            root_left = find(parent, left)
            root_right = find(parent, right)

            if root_left == root_right:
                continue

            parent[root_left] = root_right
            degree[left] += 1.0
            degree[right] += 1.0
            chosen[position] = True
            chosen_count += 1

            if chosen_count == n - 1:
                break

        frequency += chosen
        degree_sum += degree

    frequency /= float(sample_count)
    mean_degree = degree_sum / float(sample_count)

    endpoint_fit = np.exp(
        -0.55
        * (
            np.abs(mean_degree[edge_i] - 2.0)
            + np.abs(mean_degree[edge_j] - 2.0)
        )
    )

    stability = 0.20 + 0.80 * frequency
    tree_prior = stability * endpoint_fit

    if symmetric_input:
        result[edge_i, edge_j] = tree_prior
        result[edge_j, edge_i] = tree_prior
    else:
        forward_rank = np.zeros_like(distances, dtype=float)
        valid_values = distances[valid]
        sorted_values = np.argsort(valid_values, kind="stable")
        directional_ranks = np.empty(sorted_values.size, dtype=float)
        directional_ranks[sorted_values] = np.arange(
            sorted_values.size,
            dtype=float,
        )
        if directional_ranks.size > 1:
            directional_ranks /= float(directional_ranks.size - 1)
        forward_rank[valid] = directional_ranks

        for position, (left, right) in enumerate(zip(edge_i, edge_j)):
            base = tree_prior[position]
            if valid[left, right]:
                result[left, right] = base * (
                    0.35 + 0.65 * np.exp(-forward_rank[left, right])
                )
            if valid[right, left]:
                result[right, left] = base * (
                    0.35 + 0.65 * np.exp(-forward_rank[right, left])
                )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = np.max(result)
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
