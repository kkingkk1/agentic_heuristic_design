import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diverse, globally aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    np.fill_diagonal(finite, False)

    pair_valid = finite | finite.T
    np.fill_diagonal(pair_valid, False)

    if not np.any(pair_valid):
        return np.zeros((n, n), dtype=float)

    nonnegative = np.maximum(raw, 0.0)
    cleaned = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T

    cleaned[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )
    cleaned[only_forward] = nonnegative[only_forward]
    cleaned[only_backward] = nonnegative[only_backward]

    valid_values = cleaned[pair_valid]
    positive = valid_values[valid_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    low = float(np.percentile(valid_values, 10.0))
    high = float(np.percentile(valid_values, 90.0))
    spread = max(high - low, 0.25 * scale, np.finfo(float).eps)

    # Robust multi-scale distance affinity.
    temperatures = np.array(
        [0.45 * spread, 1.0 * spread, 2.25 * spread],
        dtype=float,
    )
    temperatures = np.maximum(temperatures, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)
    for temperature in temperatures:
        component = np.exp(
            -np.minimum(cleaned, high + 8.0 * scale) / temperature
        )
        component[~pair_valid] = 0.0
        np.fill_diagonal(component, 0.0)
        component_max = float(np.max(component))
        if component_max > 0.0 and np.isfinite(component_max):
            component /= component_max
        affinity += component

    affinity /= float(len(temperatures))
    affinity = 0.5 * (affinity + affinity.T)

    # Stable rank preference, useful when absolute distances have outliers.
    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid_indices = np.flatnonzero(pair_valid[i])
        if valid_indices.size == 0:
            continue

        order = valid_indices[
            np.argsort(cleaned[i, valid_indices], kind="stable")
        ]
        ranks = np.empty(order.size, dtype=float)
        ranks[np.argsort(order, kind="stable")] = np.arange(order.size)
        rank_score[i, order] = 1.0 / np.sqrt(1.0 + ranks)

    rank_score = np.sqrt(rank_score * rank_score.T)
    rank_max = float(np.max(rank_score))
    if rank_max > 0.0:
        rank_score /= rank_max

    local_score = 0.68 * affinity + 0.32 * rank_score
    local_score[~pair_valid] = 0.0

    # Global diversity: reduce popularity bias while retaining strong edges.
    row_mass = local_score.sum(axis=1)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    rarity = 1.0 / np.sqrt(row_mass[:, None] * row_mass[None, :])
    rarity /= max(float(np.max(rarity)), np.finfo(float).eps)

    diverse_score = local_score * (0.58 + 0.42 * rarity)
    diverse_score[~pair_valid] = 0.0

    # Long-term potential from promising two-hop continuations.
    normalized_affinity = affinity.copy()
    row_norm = normalized_affinity.sum(axis=1, keepdims=True)
    row_norm = np.maximum(row_norm, np.finfo(float).eps)
    normalized_affinity /= row_norm

    two_hop = normalized_affinity @ normalized_affinity
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop[~pair_valid] = 0.0
    two_hop_max = float(np.max(two_hop))
    if two_hop_max > 0.0:
        two_hop /= two_hop_max

    # Edge indispensability: retain edges whose best detour is expensive.
    detour = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        candidate = cleaned[:, [k]] + cleaned[[k], :]
        candidate[~pair_valid] = np.inf
        detour = np.minimum(detour, candidate)

    detour_advantage = np.maximum(detour - cleaned, 0.0)
    detour_advantage[~pair_valid] = 0.0
    indispensability = 1.0 - np.exp(-detour_advantage / max(spread, scale))
    indispensability[~pair_valid] = 0.0

    long_term = (
        0.42 * two_hop
        + 0.34 * indispensability
        + 0.24 * local_score
    )
    long_term[~pair_valid] = 0.0

    # Protect connectivity-critical edges using a minimum spanning forest.
    forest_edges = np.zeros((n, n), dtype=float)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue

        component = np.flatnonzero(pair_valid[start] | (np.arange(n) == start))
        if component.size == 0:
            continue

        visited[start] = True
        best_cost = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best_cost[:] = np.inf

        neighbors = np.flatnonzero(pair_valid[start])
        best_cost[neighbors] = cleaned[start, neighbors]
        parent[neighbors] = start

        for _ in range(n - 1):
            candidate_cost = best_cost.copy()
            candidate_cost[visited] = np.inf
            next_node = int(np.argmin(candidate_cost))

            if not np.isfinite(candidate_cost[next_node]):
                break

            visited[next_node] = True
            previous = parent[next_node]
            if previous >= 0:
                forest_edges[next_node, previous] = 1.0
                forest_edges[previous, next_node] = 1.0

            next_neighbors = np.flatnonzero(pair_valid[next_node] & ~visited)
            improved = cleaned[next_node, next_neighbors] < best_cost[next_neighbors]
            update_nodes = next_neighbors[improved]
            best_cost[update_nodes] = cleaned[next_node, update_nodes]
            parent[update_nodes] = next_node

    forest_edges[~pair_valid] = 0.0

    combined = (
        0.52 * diverse_score
        + 0.34 * long_term
        + 0.14 * forest_edges
    )
    combined[~pair_valid] = 0.0
    combined = 0.5 * (combined + combined.T)

    # Mild balancing prevents hub domination without erasing rare critical edges.
    balanced = combined.copy()
    for _ in range(8):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        scale_factor = np.sqrt(row_sum)
        balanced /= scale_factor
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~pair_valid] = 0.0

    result = 0.72 * combined + 0.28 * balanced
    result[~pair_valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
