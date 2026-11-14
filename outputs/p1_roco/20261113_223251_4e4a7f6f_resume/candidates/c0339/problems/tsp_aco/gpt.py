import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = ~diagonal & (finite | finite.T)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    left_only = finite & ~finite.T
    right_only = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[left_only] = raw[left_only]
    cleaned[right_only] = raw.T[right_only]
    cleaned = np.maximum(cleaned, 0.0)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed))
    penalty = maximum + 16.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    distances = np.maximum(0.5 * (distances + distances.T), 0.0)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps
    temperature = max(0.20 * scale, eps)

    affinity = np.exp(-np.minimum(distances / temperature, 60.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)
    affinity /= max(float(np.max(affinity)), eps)

    # Candidate graph: retain a small adaptive neighborhood for each node.
    candidate_count = int(np.clip(8 + 2 * np.sqrt(float(n)), 12, 32))
    candidate_count = min(candidate_count, max(1, n - 1))

    masked_distances = distances.copy()
    masked_distances[~valid] = penalty
    np.fill_diagonal(masked_distances, penalty)

    nearest_indices = np.argpartition(
        masked_distances,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), candidate_count)
    cols = nearest_indices.reshape(-1)
    candidate_mask[rows, cols] = True
    candidate_mask |= candidate_mask.T
    candidate_mask &= valid

    candidate_edges = candidate_mask & np.triu(np.ones((n, n), dtype=bool), 1)
    edge_i, edge_j = np.nonzero(candidate_edges)
    edge_cost = distances[edge_i, edge_j]

    # Local endpoint competition from the two nearest valid neighbors.
    nearest = np.min(masked_distances, axis=1)
    second = np.partition(masked_distances, 1, axis=1)[:, 1]
    local_scale = np.maximum(second - nearest, 0.14 * scale)
    local_scale = np.maximum(local_scale, eps)

    nearest_mask = masked_distances <= nearest[:, None] + eps
    gap = np.where(
        nearest_mask,
        second[:, None] - masked_distances,
        nearest[:, None] - masked_distances,
    )
    endpoint = np.exp(
        np.clip(gap / local_scale[:, None], -24.0, 24.0)
    )
    endpoint[~valid] = 0.0
    endpoint = np.sqrt(endpoint * endpoint.T)

    # Restrict two-hop bypass computation to candidate intermediates.
    local_neighbors = nearest_indices
    local_neighbor_costs = np.take_along_axis(
        masked_distances,
        local_neighbors,
        axis=1,
    )

    irreplaceability = np.zeros((n, n), dtype=float)
    for start in range(n):
        neighbors = local_neighbors[start]
        first_cost = local_neighbor_costs[start]
        two_hop = (
            first_cost[:, None]
            + masked_distances[neighbors, :]
        ).min(axis=0)

        advantage = np.maximum(two_hop - distances[start], 0.0)
        irreplaceability[start] = 1.0 - np.exp(
            -advantage / max(0.58 * scale, eps)
        )

    irreplaceability[~valid] = 0.0
    irreplaceability = 0.5 * (
        irreplaceability + irreplaceability.T
    )

    # Lightweight perturbed connectivity stability on the candidate graph.
    stability = np.zeros(edge_i.size, dtype=float)
    if edge_i.size:
        trials = int(np.clip(8 + np.sqrt(float(n)), 10, 24))
        edge_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)
        normalized_cost = edge_cost / max(scale, eps)

        def find(parent, node):
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != node:
                parent[node] = root
                node = parent[node]
            return root

        for trial in range(trials):
            phase = edge_code * 1e-6 + (trial + 1) * 1.61803398875
            perturbed = normalized_cost * (
                1.0 + 0.12 * np.sin(phase)
            )
            order = np.argsort(perturbed, kind="stable")

            parent = np.arange(n, dtype=int)
            rank = np.zeros(n, dtype=np.int8)
            selected = 0

            for index in order:
                left = int(edge_i[index])
                right = int(edge_j[index])
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

                stability[index] += 1.0
                selected += 1
                if selected == n - 1:
                    break

        stability /= float(trials)

    mst_score = np.zeros((n, n), dtype=float)
    if edge_i.size:
        mst_score[edge_i, edge_j] = stability
        mst_score[edge_j, edge_i] = stability

    score = (
        0.56 * affinity
        + 0.14 * endpoint
        + 0.24 * irreplaceability
        + 0.06 * mst_score
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()
    for _ in range(12):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.82 * score + 0.18 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
