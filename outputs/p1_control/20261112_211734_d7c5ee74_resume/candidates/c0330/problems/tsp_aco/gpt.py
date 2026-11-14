import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized TSP edge priorities."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & off_diagonal
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 75.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(0.5 * (repaired + repaired.T), 0.0)
    np.fill_diagonal(repaired, np.inf)

    eps = np.finfo(float).eps
    global_scale = float(np.median(positive))
    global_scale = max(global_scale, eps)

    usable = np.where(symmetric_valid, repaired, np.inf)

    # Build a minimum spanning forest with Prim-style updates.
    forest = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)

    for root in range(n):
        if visited[root]:
            continue

        visited[root] = True
        best_distance = usable[root].copy()
        best_parent = np.full(n, root, dtype=int)
        best_distance[root] = np.inf

        while True:
            candidates = np.where(~visited, best_distance, np.inf)
            node = int(np.argmin(candidates))
            edge_distance = float(candidates[node])

            if not np.isfinite(edge_distance):
                break

            parent = int(best_parent[node])
            if parent != node and symmetric_valid[node, parent]:
                forest[node, parent] = True
                forest[parent, node] = True

            visited[node] = True

            improved = (~visited) & (usable[node] < best_distance)
            best_distance[improved] = usable[node, improved]
            best_parent[improved] = node

    # A smooth global preference for short edges.
    global_affinity = np.exp(
        -np.clip(repaired / global_scale, 0.0, 40.0)
    )

    # Measure how strongly each endpoint is connected to the rest of the graph.
    finite_repaired = np.where(np.isfinite(repaired), repaired, fallback)
    connection_mass = np.sum(
        np.exp(-np.clip(finite_repaired / global_scale, 0.0, 40.0))
        * off_diagonal,
        axis=1,
    )
    connection_mass = np.maximum(connection_mass, eps)

    # Prefer edges touching comparatively isolated endpoints.
    isolation = 1.0 / np.sqrt(connection_mass)
    isolation = isolation / max(float(np.max(isolation)), eps)
    endpoint_protection = 1.0 + 0.45 * np.sqrt(
        isolation[:, None] * isolation[None, :]
    )

    # Cover the two strongest incident connections without using full rankings.
    nearest = np.argmin(usable, axis=1)
    nearest_distance = usable[np.arange(n), nearest]

    masked_for_second = usable.copy()
    masked_for_second[np.arange(n), nearest] = np.inf
    second_nearest_distance = np.min(masked_for_second, axis=1)
    second_nearest_distance = np.where(
        np.isfinite(second_nearest_distance),
        second_nearest_distance,
        nearest_distance,
    )

    coverage = (
        np.exp(-np.clip(repaired / (nearest_distance[:, None] + global_scale * 0.05), 0.0, 40.0))
        + np.exp(-np.clip(repaired / (second_nearest_distance[:, None] + global_scale * 0.05), 0.0, 40.0))
    )
    coverage = np.sqrt(np.maximum(coverage, 0.0))
    coverage /= max(float(np.max(coverage)), eps)

    forest_bonus = np.where(forest, 1.55, 1.0)

    score = (
        global_affinity
        * endpoint_protection
        * (0.80 + 0.35 * coverage)
        * forest_bonus
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
