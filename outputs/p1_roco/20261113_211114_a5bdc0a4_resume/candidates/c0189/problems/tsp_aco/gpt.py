import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced edge priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & offdiag
    valid = finite | finite.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    nonnegative = np.maximum(raw, 0.0)
    distance = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    distance[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )
    distance[forward_only] = nonnegative[forward_only]
    distance[backward_only] = nonnegative[backward_only]
    distance = 0.5 * (distance + distance.T)
    distance[~valid] = 0.0
    np.fill_diagonal(distance, 0.0)

    observed = distance[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    low = float(np.percentile(observed, 10.0))
    high = float(np.percentile(observed, 90.0))
    spread = max(high - low, 0.25 * scale, np.finfo(float).eps)

    temperatures = np.maximum(
        np.array([0.55, 1.15, 2.0], dtype=float) * spread,
        np.finfo(float).eps,
    )

    affinity = np.zeros((n, n), dtype=float)
    clipped_distance = np.minimum(distance, high + 8.0 * scale)

    for temperature in temperatures:
        component = np.exp(-clipped_distance / temperature)
        component[~valid] = 0.0
        np.fill_diagonal(component, 0.0)
        maximum = float(np.max(component))
        if np.isfinite(maximum) and maximum > 0.0:
            component /= maximum
        affinity += component

    affinity /= float(temperatures.size)
    affinity = 0.5 * (affinity + affinity.T)

    nearest = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)
    nearest_index = np.full(n, -1, dtype=int)

    for i in range(n):
        row = distance[i].copy()
        row[~valid[i]] = np.inf
        row[i] = np.inf
        order = np.argsort(row, kind="stable")
        finite_order = order[np.isfinite(row[order])]

        if finite_order.size:
            nearest[i] = row[finite_order[0]]
            nearest_index[i] = int(finite_order[0])
        if finite_order.size > 1:
            second[i] = row[finite_order[1]]

    local_scale = np.maximum(second - nearest, 0.20 * scale)
    local_scale[~np.isfinite(local_scale)] = 0.20 * scale
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint = np.zeros((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            endpoint[i, indices] = np.exp(
                np.clip(
                    (second[i] - distance[i, indices]) / local_scale[i],
                    -12.0,
                    12.0,
                )
            )

    endpoint = np.sqrt(endpoint * endpoint.T)
    endpoint[~valid] = 0.0

    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest_index):
        if j >= 0 and nearest_index[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    k = min(8, max(1, n - 1))
    neighbors = np.full((n, k), -1, dtype=int)

    for i in range(n):
        row = distance[i].copy()
        row[~valid[i]] = np.inf
        row[i] = np.inf
        order = np.argsort(row, kind="stable")
        finite_order = order[np.isfinite(row[order])]
        count = min(k, finite_order.size)
        if count:
            neighbors[i, :count] = finite_order[:count]

    best_bypass = np.full((n, n), np.inf, dtype=float)

    for t in range(k):
        middles = neighbors[:, t]
        for middle in middles[middles >= 0]:
            left_valid = valid[:, middle][:, None]
            right_valid = valid[middle, :][None, :]
            feasible = left_valid & right_valid
            route = distance[:, middle][:, None] + distance[middle, :][None, :]
            best_bypass = np.minimum(
                best_bypass,
                np.where(feasible, route, np.inf),
            )

    bypass_gain = np.maximum(best_bypass - distance, 0.0)
    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[valid] = 1.0 - np.exp(
        -bypass_gain[valid] / max(0.85 * scale, np.finfo(float).eps)
    )

    base = affinity / max(float(np.max(affinity)), np.finfo(float).eps)

    local_score = (
        0.58 * base
        + 0.25 * endpoint
        + 0.11 * irreplaceability
        + 0.06 * mutual
    )
    local_score[~valid] = 0.0
    local_score = 0.5 * (local_score + local_score.T)

    row_mass = local_score.sum(axis=1)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    rarity = 1.0 / np.sqrt(row_mass[:, None] * row_mass[None, :])
    rarity /= max(float(np.max(rarity)), np.finfo(float).eps)

    diverse = local_score * (0.84 + 0.16 * rarity)
    diverse[~valid] = 0.0

    forest = np.zeros((n, n), dtype=float)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue

        visited[start] = True
        best_cost = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)

        neighbors_start = np.flatnonzero(valid[start])
        best_cost[neighbors_start] = distance[start, neighbors_start]
        parent[neighbors_start] = start

        for _ in range(n - 1):
            candidate = best_cost.copy()
            candidate[visited] = np.inf
            node = int(np.argmin(candidate))

            if not np.isfinite(candidate[node]):
                break

            visited[node] = True
            previous = parent[node]

            if previous >= 0:
                forest[node, previous] = 1.0
                forest[previous, node] = 1.0

            next_neighbors = np.flatnonzero(valid[node] & ~visited)
            improved = distance[node, next_neighbors] < best_cost[next_neighbors]
            updates = next_neighbors[improved]
            best_cost[updates] = distance[node, updates]
            parent[updates] = node

    forest[~valid] = 0.0

    combined = (
        0.72 * diverse
        + 0.20 * irreplaceability
        + 0.08 * forest
    )
    combined[~valid] = 0.0
    combined = 0.5 * (combined + combined.T)

    degree = combined.sum(axis=1)
    positive_degree = degree[degree > 0.0]
    reference = (
        float(np.median(positive_degree))
        if positive_degree.size
        else 1.0
    )
    correction = np.sqrt(
        reference / np.maximum(degree[:, None], np.finfo(float).eps)
    )
    corrected = combined * correction * correction.T
    corrected[~valid] = 0.0

    result = 0.84 * combined + 0.16 * corrected
    result[~valid] = 0.0
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
