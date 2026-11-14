import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic structure-aware edge priors for stochastic TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = (finite_nonnegative | finite_nonnegative.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite_nonnegative & finite_nonnegative.T
    forward_only = finite_nonnegative & ~finite_nonnegative.T
    backward_only = ~finite_nonnegative & finite_nonnegative.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]
    cleaned = np.maximum(0.5 * (cleaned + cleaned.T), 0.0)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    if not np.isfinite(maximum):
        maximum = scale

    eps = np.finfo(float).eps
    penalty = maximum + 16.0 * scale

    distances = np.where(valid, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    temperature = max(0.30 * scale, eps)
    affinity = np.exp(
        -np.minimum(distances / temperature, 60.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    affinity /= affinity_max

    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1]
    local_scale = np.maximum(
        second - nearest,
        0.16 * scale,
    )
    local_scale = np.maximum(local_scale, eps)

    local_preference = np.exp(
        np.clip(
            -(distances - nearest[:, None])
            / local_scale[:, None],
            -24.0,
            0.0,
        )
    )
    local_preference[~valid] = 0.0
    np.fill_diagonal(local_preference, 0.0)

    local_max = np.maximum(
        np.max(local_preference, axis=1, keepdims=True),
        eps,
    )
    local_preference /= local_max

    mutual_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    regret = np.exp(
        np.clip(
            (second[:, None] - distances)
            / local_scale[:, None],
            -20.0,
            20.0,
        )
    )
    regret[~valid] = 0.0
    np.fill_diagonal(regret, 0.0)

    mutual_regret = np.sqrt(
        np.maximum(regret, 0.0)
        * np.maximum(regret.T, 0.0)
    )

    bypass = np.full((n, n), penalty, dtype=float)

    for middle in range(n):
        candidate = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        bypass = np.minimum(bypass, candidate)

    bypass = np.minimum(bypass, bypass.T)
    np.fill_diagonal(bypass, penalty)

    bypass_gain = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / max(0.72 * scale, eps)
    )
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]

    mst_stability = np.zeros((n, n), dtype=float)

    if edge_i.size:
        edge_cost = distances[edge_i, edge_j]
        edge_count = edge_i.size
        trials = int(np.clip(
            10 + 2 * np.sqrt(float(n)),
            12,
            32,
        ))
        counts = np.zeros(edge_count, dtype=float)

        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)

        def find(parent: np.ndarray, node: int) -> int:
            root = node
            while parent[root] != root:
                root = int(parent[root])

            while parent[node] != node:
                next_node = int(parent[node])
                parent[node] = root
                node = next_node

            return root

        normalized_cost = edge_cost / max(scale, eps)

        for trial in range(trials):
            phase = (
                pair_code * 1e-6
                + (trial + 1) * 1.61803398875
            )
            perturbed_cost = normalized_cost * (
                1.0 + 0.20 * np.sin(phase)
            )

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

                counts[index] += 1.0
                selected += 1

                if selected == n - 1:
                    break

        stability = counts / float(trials)
        mst_stability[edge_i, edge_j] = stability
        mst_stability[edge_j, edge_i] = stability

        stability_max = float(np.max(mst_stability))
        if np.isfinite(stability_max) and stability_max > 0.0:
            mst_stability /= stability_max

    score = (
        0.43 * affinity
        + 0.24 * mutual_preference
        + 0.15 * mutual_regret
        + 0.12 * irreplaceability
        + 0.06 * mst_stability
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    balanced = score.copy()

    for _ in range(32):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
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
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
