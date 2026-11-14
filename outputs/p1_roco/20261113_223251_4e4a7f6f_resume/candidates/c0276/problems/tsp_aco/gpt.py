import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic symmetric edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)

    valid = (~diagonal) & (
        finite_nonnegative | finite_nonnegative.T
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite_nonnegative & finite_nonnegative.T
    forward_only = finite_nonnegative & ~finite_nonnegative.T
    backward_only = ~finite_nonnegative & finite_nonnegative.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_observed = observed[np.isfinite(observed)]
    largest = (
        float(np.max(finite_observed))
        if finite_observed.size
        else scale
    )
    penalty = largest + 16.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, penalty)
    distances = 0.5 * (distances + distances.T)

    eps = np.finfo(float).eps
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

    local_best = np.min(distances, axis=1)
    local_second = np.partition(distances, 1, axis=1)[:, 1]
    local_gap = np.maximum(
        local_second - local_best,
        0.16 * scale,
    )
    local_gap = np.maximum(local_gap, eps)

    relative = np.exp(
        np.clip(
            -(distances - local_best[:, None]) / local_gap[:, None],
            -24.0,
            0.0,
        )
    )
    relative[~valid] = 0.0
    np.fill_diagonal(relative, 0.0)

    relative_max = np.maximum(
        np.max(relative, axis=1, keepdims=True),
        eps,
    )
    relative /= relative_max

    endpoint_exclusivity = np.sqrt(
        np.maximum(relative, 0.0)
        * np.maximum(relative.T, 0.0)
    )

    local_support = np.sum(relative, axis=1)
    scarcity = 1.0 / np.sqrt(
        np.maximum(local_support, eps)
    )
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.55, 1.85)

    endpoint_scarcity = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    two_hop = np.full((n, n), penalty, dtype=float)
    for middle in range(n):
        route_via_middle = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        two_hop = np.minimum(two_hop, route_via_middle)

    np.fill_diagonal(two_hop, penalty)

    bypass_gain = np.maximum(two_hop - distances, 0.0)
    bypass_scale = max(0.72 * scale, eps)
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / bypass_scale
    )
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    edge_i, edge_j = np.triu_indices(n, k=1)
    keep = valid[edge_i, edge_j]
    edge_i = edge_i[keep]
    edge_j = edge_j[keep]

    mst_stability = np.zeros((n, n), dtype=float)

    if edge_i.size:
        edge_cost = distances[edge_i, edge_j]
        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)

        trials = int(np.clip(
            12 + 2.0 * np.sqrt(float(n)),
            14,
            36,
        ))
        counts = np.zeros(edge_i.size, dtype=float)

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
                1.0 + 0.22 * np.sin(phase)
            )
            order = np.lexsort(
                (edge_j, edge_i, perturbed_cost)
            )

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

                counts[index] += 1.0
                selected += 1

                if selected == n - 1:
                    break

        stability = counts / float(max(trials, 1))
        mst_stability[edge_i, edge_j] = stability
        mst_stability[edge_j, edge_i] = stability

    mst_max = float(np.max(mst_stability))
    if np.isfinite(mst_max) and mst_max > 0.0:
        mst_stability /= mst_max

    score = (
        0.38 * affinity
        + 0.24 * endpoint_exclusivity
        + 0.16 * irreplaceability
        + 0.12 * mst_stability
        + 0.10 * endpoint_scarcity
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()
    for _ in range(40):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.68 * score + 0.32 * balanced
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
    ).astype(float, copy=False)
