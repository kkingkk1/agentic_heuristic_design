import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(symmetric, np.inf)

    positive = symmetric[np.isfinite(symmetric) & (symmetric > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = symmetric[valid]
    largest = float(np.max(finite_values)) if finite_values.size else scale
    penalty = largest + 18.0 * scale
    distances = np.where(np.isfinite(symmetric), symmetric, penalty)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps * max(scale, 1.0)

    # Global distance affinity.
    temperature = max(0.55 * scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )

    maximum = float(np.max(affinity))
    if maximum > 0.0 and np.isfinite(maximum):
        affinity /= maximum

    # Mutual local preference and endpoint competition.
    local_temperature = max(0.75 * scale, eps)
    local_weight = np.zeros((n, n), dtype=float)
    local_weight[valid] = np.exp(
        -np.minimum(distances[valid] / local_temperature, 60.0)
    )

    row_mass = np.maximum(local_weight.sum(axis=1), eps)
    row_probability = local_weight / row_mass[:, None]

    rank_order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        rank_order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(0.8 * np.sqrt(float(n)), 1.0)
    rank_preference = np.exp(-ranks / rank_scale)
    rank_preference[~valid] = 0.0

    mutual_preference = np.sqrt(
        np.maximum(
            row_probability * row_probability.T,
            0.0,
        )
    )

    endpoint_competition = np.sqrt(
        np.maximum(
            rank_preference * rank_preference.T,
            0.0,
        )
    )

    # Node scarcity: emphasize edges incident to locally constrained nodes.
    weighted_degree = np.maximum(local_weight.sum(axis=1), eps)
    scarcity = 1.0 / np.sqrt(weighted_degree)
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.55, 1.9)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )

    # Two-hop bypass advantage.
    best_bypass = np.full((n, n), penalty, dtype=float)
    for middle in range(n):
        candidate = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        best_bypass = np.minimum(best_bypass, candidate)

    np.fill_diagonal(best_bypass, penalty)
    bypass_gain = np.maximum(best_bypass - distances, 0.0)

    bypass_scale = max(0.8 * scale, eps)
    irreplaceability = 1.0 - np.exp(
        -np.minimum(bypass_gain / bypass_scale, 60.0)
    )
    irreplaceability[~valid] = 0.0

    # Deterministic perturbed-MST stability.
    upper_i, upper_j = np.triu_indices(n, k=1)
    edge_mask = valid[upper_i, upper_j]
    edge_i = upper_i[edge_mask]
    edge_j = upper_j[edge_mask]
    edge_distance = distances[edge_i, edge_j]

    edge_count = edge_i.size
    mst_stability = np.zeros((n, n), dtype=float)

    if edge_count > 0:
        normalized_cost = edge_distance / max(scale, eps)
        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)

        trials = int(np.clip(10 + 2 * np.sqrt(float(n)), 14, 28))
        edge_counts = np.zeros(edge_count, dtype=float)

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
                pair_code * 1e-6
                + (trial + 1) * 1.61803398875
            )
            perturbation = 1.0 + 0.20 * np.sin(phase)
            costs = normalized_cost * perturbation
            order = np.lexsort((edge_j, edge_i, costs))

            parent = np.arange(n, dtype=int)
            rank = np.zeros(n, dtype=np.int8)
            selected = 0

            for edge_index in order:
                left = int(edge_i[edge_index])
                right = int(edge_j[edge_index])

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

                edge_counts[edge_index] += 1.0
                selected += 1

                if selected == n - 1:
                    break

        stability = edge_counts / float(max(trials, 1))
        mst_stability[edge_i, edge_j] = stability
        mst_stability[edge_j, edge_i] = stability

    # Unified structural fusion.
    score = (
        0.30 * affinity
        + 0.18 * mutual_preference
        + 0.16 * endpoint_competition
        + 0.16 * irreplaceability
        + 0.12 * mst_stability
        + 0.08 * endpoint_scarcity
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)

    # Symmetric balancing reduces domination by high-degree nodes.
    balanced = score.copy()
    for _ in range(24):
        row_sum = np.maximum(balanced.sum(axis=1, keepdims=True), eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.62 * score + 0.38 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

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
