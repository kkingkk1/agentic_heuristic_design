import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)

    # Treat an edge as valid when at least one directed entry is finite.
    valid = (~diagonal) & (finite | finite.T)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    only_left = finite & ~finite.T
    only_right = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[only_left] = raw[only_left]
    cleaned[only_right] = raw.T[only_right]

    cleaned = np.nan_to_num(
        cleaned,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        q25, median, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(median)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    safe_scale = max(scale, eps)
    maximum = float(np.max(observed)) if observed.size else safe_scale
    if not np.isfinite(maximum):
        maximum = safe_scale

    penalty = maximum + 16.0 * safe_scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    # Direct distance affinity.
    temperature = max(
        0.22 * float(np.median(observed)) if observed.size else safe_scale,
        0.12 * safe_scale,
        eps,
    )
    affinity = np.exp(
        -np.minimum(distances / temperature, 60.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > 0.0:
        affinity /= affinity_max

    # Mutual local preference and endpoint exclusivity.
    row_sorted = np.sort(distances, axis=1)
    nearest = row_sorted[:, 0]
    second = row_sorted[:, min(1, n - 1)]

    local_gap = np.maximum(
        second - nearest,
        0.14 * safe_scale,
    )
    local_gap = np.maximum(local_gap, eps)

    local_preference = np.exp(
        np.clip(
            -(distances - nearest[:, None]) / local_gap[:, None],
            -30.0,
            0.0,
        )
    )
    local_preference[~valid] = 0.0
    np.fill_diagonal(local_preference, 0.0)

    preference_max = np.maximum(
        np.max(local_preference, axis=1, keepdims=True),
        eps,
    )
    local_preference /= preference_max

    mutual_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    row_exclusivity = 1.0 - np.exp(
        -np.maximum(second[:, None] - distances, 0.0)
        / local_gap[:, None]
    )
    col_exclusivity = 1.0 - np.exp(
        -np.maximum(second[None, :] - distances, 0.0)
        / local_gap[None, :]
    )

    exclusivity = np.sqrt(
        np.maximum(row_exclusivity, 0.0)
        * np.maximum(col_exclusivity, 0.0)
    )
    exclusivity[~valid] = 0.0
    np.fill_diagonal(exclusivity, 0.0)

    # Mutual rank signal.
    row_order = np.argsort(distances, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_score = np.exp(-row_rank / rank_scale)
    mutual_rank = np.sqrt(
        np.maximum(rank_score, 0.0)
        * np.maximum(rank_score.T, 0.0)
    )
    mutual_rank[~valid] = 0.0
    np.fill_diagonal(mutual_rank, 0.0)

    # Approximate two-hop irreplaceability through nearest intermediates.
    candidate_count = int(
        np.clip(10 + 2 * np.sqrt(float(n)), 10, 48)
    )
    candidate_count = min(candidate_count, max(1, n - 1))

    neighbor_source = distances.copy()
    np.fill_diagonal(neighbor_source, penalty)

    neighbor_ids = np.argpartition(
        neighbor_source,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    bypass = np.full((n, n), penalty, dtype=float)

    for start in range(n):
        intermediates = neighbor_ids[start]
        via = (
            distances[start, intermediates, None]
            + distances[intermediates, :]
        )
        bypass[start] = np.min(via, axis=0)

    bypass = np.minimum(bypass, bypass.T)
    np.fill_diagonal(bypass, penalty)

    bypass_gain = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / max(0.72 * safe_scale, eps)
    )
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    # Perturbed Kruskal stability as a global connectivity signal.
    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]
    edge_cost = distances[edge_i, edge_j]

    stability = np.zeros(edge_i.size, dtype=float)

    if edge_i.size:
        pair_code = (
            ((edge_i.astype(np.int64) + 1) * 73856093)
            ^ ((edge_j.astype(np.int64) + 1) * 19349663)
            ^ ((n + 1) * 83492791)
        ).astype(float)

        trials = int(np.clip(10 + 1.6 * np.sqrt(float(n)), 14, 32))
        normalized_cost = edge_cost / safe_scale

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
            perturbation = 1.0 + 0.17 * np.sin(phase)
            perturbed_cost = normalized_cost * perturbation

            order = np.lexsort(
                (edge_j, edge_i, perturbed_cost)
            )

            parent = np.arange(n, dtype=np.int32)
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

    # Fuse local, structural, and global evidence.
    local_signal = (
        0.52 * mutual_preference
        + 0.28 * mutual_rank
        + 0.20 * exclusivity
    )
    structural_signal = (
        0.68 * irreplaceability
        + 0.32 * mst_score
    )

    score = (
        0.46 * affinity
        + 0.27 * local_signal
        + 0.19 * structural_signal
        + 0.08 * mst_score
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    # Symmetric iterative degree balancing.
    balanced = score.copy()
    for _ in range(28):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    result_max = float(np.max(result))
    if np.isfinite(result_max) and result_max > 0.0:
        result /= result_max

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
