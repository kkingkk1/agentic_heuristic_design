import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    nonnegative = np.maximum(raw, 0.0)
    cleaned = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    cleaned[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )
    cleaned[forward_only] = nonnegative[forward_only]
    cleaned[backward_only] = nonnegative.T[backward_only]
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]

    scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    largest = float(np.max(observed)) if observed.size else safe_scale
    penalty = largest + 16.0 * safe_scale

    distances = np.full(
        (n, n),
        penalty,
        dtype=float,
    )
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, penalty)
    distances = 0.5 * (distances + distances.T)

    temperature = max(
        0.34 * safe_scale,
        eps,
    )
    affinity = np.exp(
        -np.minimum(distances / temperature, 60.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        return valid.astype(float)

    affinity /= affinity_max

    row_min = np.min(distances, axis=1)
    row_second = np.partition(
        distances,
        1,
        axis=1,
    )[:, 1]

    local_scale = np.maximum(
        row_second - row_min,
        0.18 * safe_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    nearest = distances <= (
        row_min[:, None] + eps
    )

    gap = np.where(
        nearest,
        row_second[:, None] - distances,
        row_min[:, None] - distances,
    )

    endpoint_score = np.exp(
        np.clip(
            gap / local_scale[:, None],
            -24.0,
            24.0,
        )
    )
    endpoint_score[~valid] = 0.0
    endpoint_score = np.sqrt(
        np.maximum(
            endpoint_score * endpoint_score.T,
            0.0,
        )
    )

    row_mass = affinity.sum(axis=1)
    preference = affinity / np.maximum(
        row_mass[:, None],
        eps,
    )
    mutual_preference = np.sqrt(
        np.maximum(
            preference * preference.T,
            0.0,
        )
    )
    mutual_preference[~valid] = 0.0

    candidate_count = int(
        np.clip(
            10.0 + 2.0 * np.sqrt(float(n)),
            10,
            40,
        )
    )
    candidate_count = min(
        candidate_count,
        max(1, n - 1),
    )

    bypass = np.full(
        (n, n),
        penalty,
        dtype=float,
    )

    for source in range(n):
        neighbors = np.flatnonzero(valid[source])
        if neighbors.size == 0:
            continue

        order = np.argsort(
            distances[source, neighbors],
            kind="stable",
        )
        intermediates = neighbors[
            order[:candidate_count]
        ]

        via = (
            distances[source, intermediates, None]
            + distances[intermediates, :]
        )
        bypass[source] = np.min(via, axis=0)

    bypass = np.minimum(
        bypass,
        bypass.T,
    )
    np.fill_diagonal(bypass, penalty)

    advantage = np.maximum(
        bypass - distances,
        0.0,
    )
    irreplaceability = 1.0 - np.exp(
        -advantage / max(0.72 * safe_scale, eps)
    )
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    upper_i, upper_j = np.triu_indices(
        n,
        k=1,
    )
    edge_mask = valid[upper_i, upper_j]
    edge_i = upper_i[edge_mask]
    edge_j = upper_j[edge_mask]

    forest_stability = np.zeros(
        edge_i.size,
        dtype=float,
    )

    if edge_i.size:
        edge_cost = distances[edge_i, edge_j]
        normalized_cost = edge_cost / safe_scale

        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)

        trials = int(
            np.clip(
                10.0 + 1.5 * np.sqrt(float(n)),
                12,
                24,
            )
        )

        def find(parent: np.ndarray, node: int) -> int:
            root = node
            while parent[root] != root:
                root = int(parent[root])

            while parent[node] != node:
                next_node = int(parent[node])
                parent[node] = root
                node = next_node

            return root

        for trial in range(trials):
            phase = (
                pair_code * 1e-6
                + (trial + 1) * 1.61803398875
            )
            perturbation = 1.0 + 0.16 * np.sin(phase)
            perturbed_cost = normalized_cost * perturbation

            order = np.lexsort(
                (
                    edge_j,
                    edge_i,
                    perturbed_cost,
                )
            )

            parent = np.arange(
                n,
                dtype=np.int32,
            )
            rank = np.zeros(
                n,
                dtype=np.int8,
            )
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

                forest_stability[index] += 1.0
                selected += 1

                if selected >= n - 1:
                    break

        forest_stability /= float(trials)

    forest_score = np.zeros(
        (n, n),
        dtype=float,
    )
    if edge_i.size:
        forest_score[edge_i, edge_j] = forest_stability
        forest_score[edge_j, edge_i] = forest_stability

    local_signal = (
        0.58 * endpoint_score
        + 0.42 * mutual_preference
    )

    score = (
        np.maximum(affinity, eps) ** 0.48
        * np.maximum(0.25 + local_signal, eps) ** 0.92
        * np.maximum(0.30 + irreplaceability, eps) ** 0.78
        * np.maximum(0.38 + forest_score, eps) ** 0.42
    )

    score[~valid] = 0.0
    score = 0.5 * (
        score + score.T
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

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
