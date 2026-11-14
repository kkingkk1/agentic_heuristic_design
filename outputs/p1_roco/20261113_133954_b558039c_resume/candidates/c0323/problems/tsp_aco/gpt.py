import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    raw = np.where(
        np.isfinite(distances) & (distances >= 0.0) & ~diagonal,
        distances,
        np.inf,
    )

    reverse = raw.T
    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)

    both_valid = forward_valid & backward_valid
    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = reverse[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    normalized_distance = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 60.0)

    direct_affinity = np.exp(-normalized_distance)

    two_hop = np.min(
        distance[:, :, None] + distance[None, :, :],
        axis=1,
    )
    two_hop[~np.isfinite(two_hop)] = np.inf

    replacement_gap = np.divide(
        two_hop - distance,
        global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid & np.isfinite(two_hop),
    )
    replacement_gap = np.clip(replacement_gap, -6.0, 6.0)

    shortcut_signal = 1.0 + 0.55 * np.tanh(replacement_gap)
    shortcut_signal[~valid] = 0.0

    conductance = np.where(
        valid,
        np.exp(-np.clip(normalized_distance, 0.0, 40.0)),
        0.0,
    )
    endpoint_strength = np.sum(conductance, axis=1)

    strength_scale = np.median(
        endpoint_strength[endpoint_strength > 0.0]
    )
    strength_scale = max(float(strength_scale), eps)

    endpoint_capacity = np.sqrt(
        strength_scale / np.maximum(endpoint_strength, eps)
    )
    endpoint_capacity = np.clip(endpoint_capacity, 0.55, 1.75)

    capacity_pair = np.sqrt(
        endpoint_capacity[:, None] * endpoint_capacity[None, :]
    )

    forest = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)

    for root in range(n):
        if visited[root]:
            continue

        visited[root] = True
        best_distance = np.full(n, np.inf, dtype=float)
        best_parent = np.full(n, -1, dtype=int)

        reachable = valid[root]
        best_distance[reachable] = distance[root, reachable]
        best_parent[reachable] = root

        for _ in range(n - 1):
            candidates = np.where(~visited & np.isfinite(best_distance))[0]
            if candidates.size == 0:
                break

            next_node = candidates[
                np.argmin(best_distance[candidates])
            ]
            parent = best_parent[next_node]

            if parent < 0:
                break

            forest[next_node, parent] = True
            forest[parent, next_node] = True
            visited[next_node] = True

            neighbors = np.where(valid[next_node] & ~visited)[0]
            improved = distance[next_node, neighbors] < best_distance[neighbors]
            improved_nodes = neighbors[improved]

            best_distance[improved_nodes] = distance[
                next_node,
                improved_nodes,
            ]
            best_parent[improved_nodes] = next_node

    forest_factor = np.where(forest, 1.65, 1.0)

    scores = (
        direct_affinity
        * shortcut_signal
        * capacity_pair
        * forest_factor
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
