import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic TSP edge priors from local ranking and cycle support."""
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
    cleaned = 0.5 * (cleaned + cleaned.T)
    cleaned[~valid] = 0.0

    values = cleaned[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    transformed = np.log1p(cleaned / safe_scale)
    transformed[~valid] = np.inf

    finite_values = transformed[valid]
    center = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - center)))
    temperature = max(1.4826 * mad, 0.18, eps)

    local_score = np.zeros((n, n), dtype=float)

    for node in range(n):
        row = transformed[node]
        available = np.flatnonzero(valid[node])

        if available.size == 0:
            continue

        local_values = row[available]
        local_order = np.argsort(
            local_values,
            kind="stable",
        )
        ranks = np.empty(available.size, dtype=float)
        ranks[local_order] = np.arange(
            available.size,
            dtype=float,
        )

        rank_fraction = ranks / max(
            float(available.size - 1),
            1.0,
        )

        row_center = float(np.median(local_values))
        relative = (
            row_center - local_values
        ) / temperature

        soft_preference = 1.0 / (
            1.0 + np.exp(
                -np.clip(relative, -30.0, 30.0)
            )
        )

        rank_preference = np.exp(
            -2.4 * rank_fraction
        )

        local_score[node, available] = (
            0.58 * soft_preference
            + 0.42 * rank_preference
        )

    local_score[~valid] = 0.0

    symmetric_local = np.sqrt(
        np.maximum(local_score, 0.0)
        * np.maximum(local_score.T, 0.0)
    )
    symmetric_local[~valid] = 0.0

    row_entropy = np.zeros(n, dtype=float)
    row_mass = local_score.sum(axis=1)

    for node in range(n):
        if row_mass[node] <= eps:
            continue

        probabilities = local_score[node] / row_mass[node]
        probabilities = probabilities[probabilities > 0.0]

        if probabilities.size:
            entropy = -float(
                np.sum(
                    probabilities
                    * np.log(np.maximum(probabilities, eps))
                )
            )
            row_entropy[node] = entropy / max(
                np.log(float(probabilities.size)),
                1.0,
            )

    scarcity = 1.0 - row_entropy
    scarcity = 0.55 + 0.90 * scarcity

    endpoint_pressure = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_pressure[~valid] = 0.0

    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]

    cycle_support = np.zeros(edge_i.size, dtype=float)

    if edge_i.size:
        base_utility = (
            0.62 * symmetric_local[edge_i, edge_j]
            + 0.23 * endpoint_pressure[edge_i, edge_j]
            + 0.15 * np.exp(
                -np.maximum(
                    transformed[edge_i, edge_j] - center,
                    0.0,
                )
                / max(temperature, eps)
            )
        )

        edge_count = edge_i.size
        trials = int(np.clip(
            10 + 2 * np.sqrt(float(n)),
            12,
            32,
        ))

        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
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
                + (trial + 1) * 1.324717957
            )

            perturbation = (
                1.0
                + 0.08 * np.sin(phase)
                + 0.04 * np.cos(phase * 0.37)
            )

            utility = base_utility * perturbation
            order = np.lexsort(
                (
                    edge_j,
                    edge_i,
                    -utility,
                )
            )

            parent = np.arange(n, dtype=np.int32)
            rank = np.zeros(n, dtype=np.int8)
            degree = np.zeros(n, dtype=np.int8)
            selected = 0

            for index in order:
                left = int(edge_i[index])
                right = int(edge_j[index])

                if degree[left] >= 2 or degree[right] >= 2:
                    continue

                root_left = find(parent, left)
                root_right = find(parent, right)

                closes_cycle = root_left == root_right

                if closes_cycle and selected < n - 1:
                    continue

                if not closes_cycle:
                    if rank[root_left] < rank[root_right]:
                        parent[root_left] = root_right
                    elif rank[root_left] > rank[root_right]:
                        parent[root_right] = root_left
                    else:
                        parent[root_right] = root_left
                        rank[root_left] += 1

                degree[left] += 1
                degree[right] += 1
                cycle_support[index] += 1.0
                selected += 1

                if selected >= n:
                    break

        cycle_support /= float(max(trials, 1))

    global_cycle = np.zeros((n, n), dtype=float)
    if edge_i.size:
        global_cycle[edge_i, edge_j] = cycle_support
        global_cycle[edge_j, edge_i] = cycle_support

    direct_quality = np.zeros((n, n), dtype=float)
    direct_quality[valid] = np.exp(
        -np.clip(
            (transformed[valid] - center)
            / max(temperature, eps),
            -30.0,
            30.0,
        )
    )

    direct_quality[valid] /= max(
        float(np.max(direct_quality[valid])),
        eps,
    )

    score = (
        0.42 * direct_quality
        + 0.30 * symmetric_local
        + 0.13 * endpoint_pressure
        + 0.15 * global_cycle
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()

    for _ in range(28):
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
        0.72 * score
        + 0.28 * balanced
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
