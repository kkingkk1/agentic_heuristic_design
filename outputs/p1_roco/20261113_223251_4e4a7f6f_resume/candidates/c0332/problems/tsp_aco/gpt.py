import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = 0.5 * (raw[valid] + raw.T[valid])
    positive = values[values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(positive, [0.25, 0.50, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    upper = max(
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.75)),
        scale,
    )
    penalty = upper + 14.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, penalty)

    normalized = np.minimum(distances / max(0.78 * scale, eps), 60.0)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + 1e-9 * scale)
    inverse /= max(float(inverse.max()), eps)

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~valid] = 0.0

    row_mass = affinity.sum(axis=1)
    normalized_affinity = affinity / np.maximum(row_mass[:, None], eps)
    mutual_preference = np.sqrt(
        np.maximum(normalized_affinity * normalized_affinity.T, 0.0)
    )
    mutual_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )
    mutual_rank[~valid] = 0.0

    sorted_distances = np.sort(distances, axis=1)
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1]
    gap_scale = np.maximum(second - nearest, 0.18 * scale)
    gap_scale = np.maximum(gap_scale, eps)

    exclusivity = np.maximum(second[:, None] - distances, 0.0)
    exclusivity = 1.0 - np.exp(-exclusivity / gap_scale[:, None])
    exclusivity[~valid] = 0.0

    mutual_exclusivity = np.sqrt(
        np.maximum(exclusivity * exclusivity.T, 0.0)
    )
    mutual_exclusivity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via = (
            distances[:, start:stop, None]
            + distances[None, start:stop, :]
        )
        bypass = np.minimum(bypass, np.min(via, axis=1))

    np.fill_diagonal(bypass, penalty)

    detour_gain = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -detour_gain / max(0.70 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    scarcity = 1.0 / np.sqrt(np.maximum(row_mass, eps))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.65, 1.75)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    endpoint_scarcity[~valid] = 0.0

    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]
    edge_cost = distances[edge_i, edge_j]

    connectivity = np.zeros(edge_i.size, dtype=float)

    if edge_i.size:
        trials = int(np.clip(10 + 2 * np.sqrt(float(n)), 14, 36))
        edge_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)

        scaled_cost = edge_cost / max(scale, eps)

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
            phase = edge_code * 1e-6 + (trial + 1) * 1.61803398875
            perturbation = 1.0 + 0.16 * np.sin(phase)
            perturbed_cost = scaled_cost * perturbation
            sequence = np.lexsort((edge_j, edge_i, perturbed_cost))

            parent = np.arange(n, dtype=int)
            tree_rank = np.zeros(n, dtype=np.int8)
            selected = 0

            for index in sequence:
                left = int(edge_i[index])
                right = int(edge_j[index])

                root_left = find(parent, left)
                root_right = find(parent, right)

                if root_left == root_right:
                    continue

                if tree_rank[root_left] < tree_rank[root_right]:
                    parent[root_left] = root_right
                elif tree_rank[root_left] > tree_rank[root_right]:
                    parent[root_right] = root_left
                else:
                    parent[root_right] = root_left
                    tree_rank[root_left] += 1

                connectivity[index] += 1.0
                selected += 1

                if selected == n - 1:
                    break

        connectivity /= float(trials)

    mst_signal = np.zeros((n, n), dtype=float)
    if edge_i.size:
        mst_signal[edge_i, edge_j] = connectivity
        mst_signal[edge_j, edge_i] = connectivity

    local_signal = (
        0.42 * mutual_preference
        + 0.33 * mutual_rank
        + 0.25 * mutual_exclusivity
    )

    structural_signal = (
        0.58 * irreplaceability
        + 0.27 * endpoint_scarcity
        + 0.15 * mst_signal
    )

    score = (
        np.maximum(affinity, eps) ** 0.82
        * np.maximum(inverse, eps) ** 0.58
        * np.maximum(0.30 + local_signal, eps) ** 0.92
        * np.maximum(0.55 + structural_signal, eps) ** 0.78
        * np.maximum(0.86 + 0.28 * mst_signal, eps)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    balanced = score.copy()
    for _ in range(30):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.70 * score + 0.30 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
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
