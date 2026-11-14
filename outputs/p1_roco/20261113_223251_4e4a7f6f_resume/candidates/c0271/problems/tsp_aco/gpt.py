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

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    upper = max(
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.75)),
        safe_scale,
    )
    replacement = upper + 14.0 * safe_scale

    filled = np.where(valid, distances, replacement)
    np.fill_diagonal(filled, replacement)

    normalized = np.minimum(
        filled / safe_scale,
        60.0,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        distances[valid] + safe_scale * 1e-9
    )
    inverse /= max(float(np.max(inverse)), eps)

    order = np.argsort(
        filled,
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
        0.85 * np.sqrt(float(n)),
    )
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~valid] = 0.0

    row_mass = affinity.sum(axis=1)
    row_preference = affinity / np.maximum(
        row_mass[:, None],
        eps,
    )
    mutual_preference = np.sqrt(
        np.maximum(
            row_preference * row_preference.T,
            eps,
        )
    )
    mutual_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(
            rank_score * rank_score.T,
            eps,
        )
    )
    mutual_rank[~valid] = 0.0

    row_sorted = np.sort(filled, axis=1)
    nearest = row_sorted[:, 0]
    second = row_sorted[:, 1]
    gap_scale = np.maximum(
        second - nearest,
        0.16 * safe_scale,
    )
    gap_scale = np.maximum(gap_scale, eps)

    relative_gap = np.maximum(
        second[:, None] - filled,
        0.0,
    )
    endpoint_exclusivity = 1.0 - np.exp(
        -relative_gap / gap_scale[:, None]
    )
    endpoint_exclusivity[~valid] = 0.0

    mutual_exclusivity = np.sqrt(
        np.maximum(
            endpoint_exclusivity
            * endpoint_exclusivity.T,
            0.0,
        )
    )
    mutual_exclusivity[~valid] = 0.0

    bypass = np.full(
        (n, n),
        replacement,
        dtype=float,
    )

    block_size = max(
        1,
        min(
            n,
            2_000_000 // max(n, 1),
        ),
    )

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        candidate = (
            filled[:, start:stop, None]
            + filled[None, start:stop, :]
        )
        bypass = np.minimum(
            bypass,
            np.min(candidate, axis=1),
        )

    np.fill_diagonal(bypass, replacement)

    detour_gain = np.maximum(
        bypass - filled,
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
    edge_cost = distances[edge_i, edge_j]

    stability = np.zeros(edge_i.size, dtype=float)

    if edge_i.size:
        trials = int(np.clip(
            12 + 2 * np.sqrt(float(n)),
            16,
            40,
        ))

        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
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
            perturbation = 1.0 + 0.18 * np.sin(phase)
            perturbed = normalized_cost * perturbation
            order = np.lexsort(
                (edge_j, edge_i, perturbed)
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

        stability /= float(trials)

    mst_stability = np.zeros((n, n), dtype=float)
    if edge_i.size:
        mst_stability[edge_i, edge_j] = stability
        mst_stability[edge_j, edge_i] = stability

    signals = (
        np.maximum(affinity, eps)
        * np.maximum(inverse, eps)
        * np.maximum(
            0.55 * mutual_rank
            + 0.45 * mutual_preference,
            eps,
        )
        * np.maximum(
            0.62 * mutual_exclusivity
            + 0.38 * irreplaceability
            + 0.16,
            eps,
        )
        * np.maximum(
            0.84 + 0.32 * mst_stability,
            eps,
        )
    )

    signals[~valid] = 0.0
    score = 0.5 * (
        signals + signals.T
    )
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    balanced = score.copy()

    for _ in range(28):
        row_sum = balanced.sum(
            axis=1,
            keepdims=True,
        )
        balanced /= np.maximum(
            row_sum,
            eps,
        )
        balanced = 0.5 * (
            balanced + balanced.T
        )
        balanced[~valid] = 0.0

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

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
