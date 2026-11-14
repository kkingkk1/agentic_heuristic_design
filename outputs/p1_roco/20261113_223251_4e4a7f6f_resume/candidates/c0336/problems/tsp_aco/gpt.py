import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T & ~diagonal
    forward_only = finite & ~finite.T & ~diagonal
    backward_only = ~finite & finite.T & ~diagonal

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

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = np.clip(
        cleaned[valid] / safe_scale,
        0.0,
        60.0,
    )
    normalized[~valid] = 60.0
    np.fill_diagonal(normalized, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-0.85 * normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        normalized[valid] + 1e-8
    )
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = affinity.sum(axis=1, keepdims=True)
    preference = affinity / np.maximum(row_mass, eps)
    mutual_preference = np.sqrt(
        np.maximum(preference * preference.T, 0.0)
    )
    mutual_preference[~valid] = 0.0

    order = np.argsort(
        normalized,
        axis=1,
        kind="stable",
    )
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(
        1.0,
        0.82 * np.sqrt(float(n)),
    )
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )
    mutual_rank[~valid] = 0.0

    sorted_distances = np.sort(normalized, axis=1)
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1]

    gap_scale = np.maximum(
        second - nearest,
        0.14,
    )
    gap_scale = np.maximum(gap_scale, eps)

    row_gap = np.maximum(
        second[:, None] - normalized,
        0.0,
    )
    row_exclusivity = 1.0 - np.exp(
        -row_gap / gap_scale[:, None]
    )
    row_exclusivity[~valid] = 0.0

    col_sorted = np.sort(normalized, axis=0)
    col_second = col_sorted[1, :]
    col_gap = np.maximum(
        col_second[None, :] - normalized,
        0.0,
    )
    col_exclusivity = 1.0 - np.exp(
        -col_gap / gap_scale[None, :]
    )
    col_exclusivity[~valid] = 0.0

    exclusivity = np.sqrt(
        np.maximum(
            row_exclusivity * col_exclusivity,
            0.0,
        )
    )
    exclusivity[~valid] = 0.0

    candidate_count = int(np.clip(
        10 + 2 * np.sqrt(float(n)),
        10,
        40,
    ))
    candidate_count = min(
        candidate_count,
        max(1, n - 1),
    )

    neighbor_ids = np.argpartition(
        normalized,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    bypass = np.full(
        (n, n),
        60.0,
        dtype=float,
    )

    for source in range(n):
        intermediates = neighbor_ids[source]
        two_hop = (
            normalized[source, intermediates, None]
            + normalized[intermediates, :]
        )
        bypass[source] = np.min(two_hop, axis=0)

    bypass = np.minimum(bypass, bypass.T)
    np.fill_diagonal(bypass, 60.0)

    bypass_gain = np.maximum(
        bypass - normalized,
        0.0,
    )
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / 0.72
    )
    irreplaceability[~valid] = 0.0

    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]
    edge_cost = normalized[edge_i, edge_j]

    stability = np.zeros(edge_i.size, dtype=float)

    if edge_i.size:
        trials = int(np.clip(
            10 + 1.5 * np.sqrt(float(n)),
            12,
            30,
        ))

        pair_code = (
            ((edge_i.astype(np.int64) + 1) * 73856093)
            ^ ((edge_j.astype(np.int64) + 1) * 19349663)
            ^ ((n + 1) * 83492791)
        ).astype(float)

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
            perturbation = 1.0 + 0.16 * np.sin(phase)
            perturbed_cost = edge_cost * perturbation

            edge_order = np.lexsort(
                (edge_j, edge_i, perturbed_cost)
            )

            parent = np.arange(n, dtype=np.int32)
            rank = np.zeros(n, dtype=np.int8)

            for index in edge_order:
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

        stability /= float(trials)

    forest_signal = np.zeros((n, n), dtype=float)
    if edge_i.size:
        forest_signal[edge_i, edge_j] = stability
        forest_signal[edge_j, edge_i] = stability

    local_signal = (
        0.56 * mutual_rank
        + 0.44 * mutual_preference
    )
    local_signal[~valid] = 0.0

    structural_signal = (
        0.58 * exclusivity
        + 0.42 * irreplaceability
    )
    structural_signal[~valid] = 0.0

    score = (
        np.maximum(affinity, eps) ** 0.95
        * np.maximum(inverse, eps) ** 0.72
        * np.maximum(local_signal, eps) ** 1.08
        * np.maximum(structural_signal + 0.18, eps) ** 0.92
        * np.maximum(0.84 + 0.34 * forest_signal, eps)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    balanced = score.copy()
    for _ in range(26):
        row_sum = balanced.sum(
            axis=1,
            keepdims=True,
        )
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (
            balanced + balanced.T
        )
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = (
        0.64 * score
        + 0.36 * balanced
    )
    result[~valid] = 0.0
    result = 0.5 * (
        result + result.T
    )
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
