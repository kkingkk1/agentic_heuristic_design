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
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

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

    observed = cleaned[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(observed))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed))
    if not np.isfinite(maximum):
        maximum = scale

    penalty = maximum + 16.0 * scale
    distances = np.where(valid, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps

    temperature = max(0.34 * scale, eps)
    affinity = np.exp(-np.minimum(distances / temperature, 60.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_support = affinity.sum(axis=1)
    support_scale = float(np.median(row_support[row_support > 0.0]))
    if not np.isfinite(support_scale) or support_scale <= 0.0:
        support_scale = 1.0

    scarcity = np.sqrt(
        np.clip(support_scale / np.maximum(row_support, eps), 0.0, 8.0)
    )
    scarcity = np.minimum(scarcity, 2.0)
    scarcity_edge = np.sqrt(scarcity[:, None] * scarcity[None, :])
    scarcity_edge[~valid] = 0.0

    row_distances = np.where(valid, distances, np.inf)
    ordered = np.sort(row_distances, axis=1)

    nearest = ordered[:, 0]
    if n > 2:
        second = ordered[:, 1]
    else:
        second = nearest

    finite_nearest = np.where(np.isfinite(nearest), nearest, scale)
    finite_second = np.where(np.isfinite(second), second, finite_nearest)

    local_gap = np.maximum(
        finite_second - finite_nearest,
        0.18 * scale,
    )
    local_gap = np.maximum(local_gap, eps)

    local_preference = np.exp(
        np.clip(
            -(distances - finite_nearest[:, None])
            / local_gap[:, None],
            -24.0,
            0.0,
        )
    )
    local_preference[~valid] = 0.0
    np.fill_diagonal(local_preference, 0.0)

    local_max = np.maximum(
        local_preference.max(axis=1, keepdims=True),
        eps,
    )
    local_preference /= local_max

    mutual_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    replacement_value = np.exp(
        np.clip(
            (finite_second[:, None] - distances)
            / local_gap[:, None],
            -20.0,
            20.0,
        )
    )
    replacement_value[~valid] = 0.0
    np.fill_diagonal(replacement_value, 0.0)

    endpoint_exclusivity = np.sqrt(
        np.maximum(replacement_value, 0.0)
        * np.maximum(replacement_value.T, 0.0)
    )

    rank_order = np.argsort(row_distances, axis=1, kind="stable")
    ranks = np.empty_like(rank_order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, rank_order[i]] = rank_values

    rank_temperature = max(2.0, np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_temperature)
    rank_score[~valid] = 0.0
    np.fill_diagonal(rank_score, 0.0)

    mutual_rank = np.sqrt(
        np.maximum(rank_score, 0.0)
        * np.maximum(rank_score.T, 0.0)
    )

    forest = np.zeros((n, n), dtype=float)
    visited = np.zeros(n, dtype=bool)

    for root in range(n):
        if visited[root]:
            continue

        visited[root] = True
        best_distance = np.full(n, np.inf, dtype=float)
        best_parent = np.full(n, -1, dtype=int)

        for node in range(n):
            if valid[root, node]:
                best_distance[node] = distances[root, node]
                best_parent[node] = root

        for _ in range(n - 1):
            candidates = np.where(~visited, best_distance, np.inf)
            next_node = int(np.argmin(candidates))

            if not np.isfinite(candidates[next_node]):
                break

            parent = int(best_parent[next_node])
            if parent < 0:
                break

            visited[next_node] = True
            forest[next_node, parent] = 1.0
            forest[parent, next_node] = 1.0

            for node in range(n):
                if visited[node] or not valid[next_node, node]:
                    continue
                candidate_distance = distances[next_node, node]
                if candidate_distance < best_distance[node]:
                    best_distance[node] = candidate_distance
                    best_parent[node] = next_node

    forest_quality = forest * np.exp(
        -distances / max(1.5 * scale, eps)
    )

    score = (
        0.34 * affinity
        + 0.22 * mutual_preference
        + 0.16 * endpoint_exclusivity
        + 0.12 * mutual_rank
        + 0.09 * scarcity_edge
        + 0.07 * forest_quality
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()

    for _ in range(24):
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

    result_max = float(np.max(result))
    if np.isfinite(result_max) and result_max > 0.0:
        result /= result_max

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
