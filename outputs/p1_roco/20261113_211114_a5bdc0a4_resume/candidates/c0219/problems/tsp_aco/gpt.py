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

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = offdiag & (finite | finite.T)

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
    penalty = maximum + 14.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, penalty)
    distances = 0.5 * (distances + distances.T)

    eps = np.finfo(float).eps
    temperature = max(0.28 * scale, eps)

    affinity = np.exp(-np.minimum(distances / temperature, 60.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)
    affinity /= max(float(np.max(affinity)), eps)

    nearest = np.min(distances, axis=1)
    second = np.partition(distances, 1, axis=1)[:, 1]
    local_scale = np.maximum(second - nearest, 0.18 * scale)
    local_scale = np.maximum(local_scale, eps)

    nearest_mask = distances <= nearest[:, None] + eps
    gap = np.where(
        nearest_mask,
        second[:, None] - distances,
        nearest[:, None] - distances,
    )
    endpoint_competition = np.exp(
        np.clip(gap / local_scale[:, None], -24.0, 24.0)
    )
    endpoint_competition[~valid] = 0.0
    endpoint_competition = np.sqrt(
        endpoint_competition * endpoint_competition.T
    )

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        candidate = (
            distances[:, start:stop, None]
            + distances[None, start:stop, :]
        )
        bypass = np.minimum(bypass, np.min(candidate, axis=1))

    np.fill_diagonal(bypass, penalty)
    advantage = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -advantage / max(0.80 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]
    edge_cost = distances[edge_i, edge_j]

    stability = np.zeros(edge_i.size, dtype=float)

    if edge_i.size:
        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)

        trials = int(np.clip(10 + 2 * np.sqrt(float(n)), 14, 36))

        def find(parent, node):
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != node:
                parent[node] = root
                node = parent[node]
            return root

        normalized_cost = edge_cost / max(scale, eps)

        for trial in range(trials):
            phase = pair_code * 1e-6 + (trial + 1) * 1.61803398875
            perturbation = 1.0 + 0.20 * np.sin(phase)
            costs = normalized_cost * perturbation
            order = np.lexsort((edge_j, edge_i, costs))

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
        0.42 * affinity
        + 0.25 * endpoint_competition
        + 0.18 * irreplaceability
        + 0.15 * mst_score
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()
    for _ in range(36):
        row_sum = np.maximum(balanced.sum(axis=1, keepdims=True), eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.64 * score + 0.36 * balanced
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
