import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic graph-aware edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    valid &= ~np.eye(n, dtype=bool)

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
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)
    safe_distance = np.where(np.isfinite(symmetric), symmetric, 0.0)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe_distance[valid] + eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    col_max = np.max(inverse, axis=0, keepdims=True)

    row_competition = inverse / np.maximum(row_max, eps)
    col_competition = inverse / np.maximum(col_max, eps)
    mutual_competition = np.sqrt(
        np.maximum(row_competition * col_competition, 0.0)
    )

    temperature = max(0.75 * scale, eps)
    smooth_distance = np.zeros((n, n), dtype=float)
    smooth_distance[valid] = np.exp(
        -np.minimum(safe_distance[valid] / temperature, 60.0)
    )

    row_mass = smooth_distance.sum(axis=1)
    col_mass = smooth_distance.sum(axis=0)
    row_mass = np.maximum(row_mass, eps)
    col_mass = np.maximum(col_mass, eps)

    mutual_probability = np.sqrt(
        np.maximum(
            (smooth_distance / row_mass[:, None])
            * (smooth_distance / col_mass[None, :]),
            0.0,
        )
    )

    base_score = inverse * (
        0.34
        + 0.28 * row_competition
        + 0.22 * col_competition
        + 0.16 * mutual_probability
    )
    base_score[~valid] = 0.0

    edge_i, edge_j = np.triu_indices(n, k=1)
    selected = valid[edge_i, edge_j]
    edge_i = edge_i[selected]
    edge_j = edge_j[selected]

    edge_count = edge_i.size
    if edge_count == 0:
        return np.zeros((n, n), dtype=float)

    edge_distance = symmetric[edge_i, edge_j]
    edge_base = edge_distance / max(scale, eps)

    pair_code = (
        (edge_i.astype(np.int64) + 1) * 73856093
        ^ (edge_j.astype(np.int64) + 1) * 19349663
        ^ (n + 1) * 83492791
    ).astype(float)

    trials = int(np.clip(10 + 2 * np.sqrt(float(n)), 12, 30))
    stability_counts = np.zeros(edge_count, dtype=float)

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
        phase = pair_code * 1e-6 + (trial + 1) * 1.61803398875
        perturbation = 1.0 + 0.20 * np.sin(phase)
        costs = edge_base * perturbation

        order = np.lexsort((edge_j, edge_i, costs))
        parent = np.arange(n, dtype=int)
        rank = np.zeros(n, dtype=np.int8)
        selected_count = 0

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

            stability_counts[edge_index] += 1.0
            selected_count += 1

            if selected_count == n - 1:
                break

    stability = stability_counts / float(trials)

    detour = np.zeros(edge_count, dtype=float)
    for index in range(edge_count):
        left = edge_i[index]
        right = edge_j[index]

        alternatives = (
            symmetric[left, :] + symmetric[:, right]
        )
        alternatives[left] = np.inf
        alternatives[right] = np.inf

        two_hop = float(np.min(alternatives))
        direct = max(float(symmetric[left, right]), eps)

        if np.isfinite(two_hop):
            detour[index] = np.tanh(
                max(two_hop / direct - 1.0, 0.0)
            )
        else:
            detour[index] = 1.0

    edge_base_score = base_score[edge_i, edge_j]
    edge_mutual = mutual_competition[edge_i, edge_j]
    edge_smooth = smooth_distance[edge_i, edge_j]

    structural_score = (
        0.36 * stability
        + 0.24 * edge_mutual
        + 0.18 * edge_smooth
        + 0.14 * detour
        + 0.08 * edge_base_score
    )

    result = np.zeros((n, n), dtype=float)
    result[edge_i, edge_j] = structural_score
    result[edge_j, edge_i] = structural_score
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

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
