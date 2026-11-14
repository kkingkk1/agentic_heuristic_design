import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return scores

    forward = np.where(raw_valid, distances, np.inf)
    backward = forward.T

    distance = np.minimum(forward, backward)
    valid = np.isfinite(distance) & ~diagonal

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    nearest = np.full(n, global_scale, dtype=float)
    second_nearest = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = distance[i]
        row_values = row[np.isfinite(row)]

        if row_values.size == 0:
            continue

        row_values = np.sort(row_values)

        nearest[i] = row_values[0]

        if row_values.size > 1:
            second_nearest[i] = row_values[1]
        else:
            second_nearest[i] = row_values[0]

    nearest = np.maximum(nearest, 0.0)
    second_nearest = np.maximum(second_nearest, nearest)

    replacement_gap = np.maximum(
        second_nearest - nearest,
        0.0,
    )

    endpoint_scale = np.sqrt(
        np.maximum(nearest, global_scale * 0.05)[:, None]
        * np.maximum(nearest, global_scale * 0.05)[None, :]
    )

    relative_length = distance / np.maximum(
        global_scale + endpoint_scale,
        eps,
    )

    length_score = 1.0 / (
        1.0
        + np.clip(relative_length, 0.0, 50.0)
    )

    endpoint_regret = replacement_gap / np.maximum(
        second_nearest + global_scale,
        eps,
    )

    edge_regret = np.sqrt(
        endpoint_regret[:, None] * endpoint_regret[None, :]
    )

    local_support = np.sqrt(
        np.maximum(nearest[:, None], eps)
        * np.maximum(nearest[None, :], eps)
    ) / np.maximum(distance + global_scale, eps)

    local_support = np.clip(local_support, 0.0, 1.0)

    # Build a minimum spanning forest with Prim's algorithm.
    forest_edges = np.zeros((n, n), dtype=bool)
    forest_degree = np.zeros(n, dtype=float)
    visited = np.zeros(n, dtype=bool)

    for root in range(n):
        if visited[root]:
            continue

        best_cost = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best_cost[root] = 0.0

        while True:
            candidate_cost = np.where(visited, np.inf, best_cost)
            current = int(np.argmin(candidate_cost))

            if not np.isfinite(candidate_cost[current]):
                break

            visited[current] = True
            previous = parent[current]

            if previous >= 0:
                forest_edges[current, previous] = True
                forest_edges[previous, current] = True
                forest_degree[current] += 1.0
                forest_degree[previous] += 1.0

            neighbors = valid[current] & ~visited
            improved = neighbors & (distance[current] < best_cost)

            best_cost[improved] = distance[current, improved]
            parent[improved] = current

    forest_bonus = 1.0 + 2.20 * forest_edges.astype(float)

    degree_penalty = 1.0 / np.sqrt(
        (
            1.0
            + 0.28 * np.maximum(forest_degree[:, None] - 2.0, 0.0)
        )
        * (
            1.0
            + 0.28 * np.maximum(forest_degree[None, :] - 2.0, 0.0)
        )
    )

    endpoint_need = 1.0 / np.sqrt(
        (
            1.0
            + 0.18 * forest_degree[:, None]
        )
        * (
            1.0
            + 0.18 * forest_degree[None, :]
        )
    )

    scores = (
        length_score
        * (0.72 + 0.28 * edge_regret)
        * (0.78 + 0.22 * local_support)
        * forest_bonus
        * degree_penalty
        * endpoint_need
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
