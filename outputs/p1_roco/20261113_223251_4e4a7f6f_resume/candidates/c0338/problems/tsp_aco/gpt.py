import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic structure-aware edge priors for stochastic TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = offdiag & (finite | finite.T)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]
    cleaned = np.maximum(cleaned, 0.0)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)
    maximum = float(np.max(observed)) if observed.size else safe_scale
    if not np.isfinite(maximum):
        maximum = safe_scale

    replacement = maximum + 16.0 * safe_scale
    distances = np.full((n, n), replacement, dtype=float)
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, replacement)
    distances = 0.5 * (distances + distances.T)

    temperature = max(0.27 * safe_scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > 0.0:
        affinity /= affinity_max

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + 1e-9 * safe_scale)
    inverse_max = float(np.max(inverse))
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        inverse /= inverse_max

    sorted_distances = np.sort(distances, axis=1)
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1]
    local_gap = np.maximum(
        second - nearest,
        0.18 * safe_scale,
    )
    local_gap = np.maximum(local_gap, eps)

    nearest_mask = distances <= nearest[:, None] + eps
    relative_gap = np.where(
        nearest_mask,
        second[:, None] - distances,
        nearest[:, None] - distances,
    )

    competition = np.exp(
        np.clip(relative_gap / local_gap[:, None], -24.0, 24.0)
    )
    competition[~valid] = 0.0
    competition = np.sqrt(
        np.maximum(competition * competition.T, 0.0)
    )
    competition[~valid] = 0.0

    support = affinity.sum(axis=1)
    scarcity = 1.0 / np.sqrt(np.maximum(support, eps))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.65, 1.8)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    bypass = np.full((n, n), replacement, dtype=float)
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via = (
            distances[:, start:stop, None]
            + distances[None, start:stop, :]
        )
        bypass = np.minimum(
            bypass,
            np.min(via, axis=1),
        )

    np.fill_diagonal(bypass, replacement)

    detour_gain = np.maximum(
        bypass - distances,
        0.0,
    )
    irreplaceability = 1.0 - np.exp(
        -detour_gain / max(0.72 * safe_scale, eps)
    )
    irreplaceability[~valid] = 0.0

    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]

    stability = np.zeros(edge_i.size, dtype=float)

    if edge_i.size:
        edge_cost = distances[edge_i, edge_j]
        trials = int(np.clip(
            10 + 2 * np.sqrt(float(n)),
            14,
            32,
        ))

        pair_code = (
            ((edge_i.astype(np.int64) + 1) * 73856093)
            ^ ((edge_j.astype(np.int64) + 1) * 19349663)
            ^ ((n + 1) * 83492791)
        ).astype(float)

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

        stability /= float(max(trials, 1))

    mst_stability = np.zeros((n, n), dtype=float)
    if edge_i.size:
        mst_stability[edge_i, edge_j] = stability
        mst_stability[edge_j, edge_i] = stability

    local_signal = (
        0.58 * competition
        + 0.42 * endpoint_scarcity
    )
    structural_signal = (
        0.68 * irreplaceability
        + 0.32 * mst_stability
    )

    score = (
        np.maximum(affinity, eps) ** 0.84
        * np.maximum(inverse, eps) ** 0.36
        * np.maximum(0.42 + local_signal, eps) ** 0.82
        * np.maximum(0.58 + structural_signal, eps) ** 0.68
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    score_max = float(np.max(score))
    if not np.isfinite(score_max) or score_max <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= score_max

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
