import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized graph-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.clip(distances[valid] / safe_scale, 0.0, 60.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + safe_scale * 1e-9)
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = np.maximum(affinity.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(affinity.sum(axis=0, keepdims=True), eps)
    row_share = affinity / row_mass
    col_share = affinity / col_mass
    mutual_competition = np.sqrt(
        np.maximum(row_share * col_share, eps)
    )
    mutual_competition[~valid] = 0.0

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = rank_values
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~valid] = 0.0
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    neighborhood = min(4, n - 1)
    local_scale = np.full(n, safe_scale, dtype=float)
    for i in range(n):
        neighbors = distances[i, valid[i]]
        if neighbors.size:
            local_scale[i] = max(
                float(np.mean(np.sort(neighbors)[:neighborhood])),
                safe_scale * 0.05,
            )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    contrast = np.zeros((n, n), dtype=float)
    contrast[valid] = np.exp(
        -np.clip(
            distances[valid] / (pair_scale[valid] + eps),
            0.0,
            60.0,
        )
    )

    detour = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not valid[i, j]:
                continue

            intermediates = np.arange(n)
            intermediates = intermediates[
                (intermediates != i) & (intermediates != j)
            ]
            if intermediates.size:
                candidates = (
                    distances[i, intermediates]
                    + distances[intermediates, j]
                )
                best_detour = float(np.min(candidates))
            else:
                best_detour = np.inf

            if np.isfinite(best_detour):
                ratio = best_detour / max(
                    distances[i, j],
                    safe_scale * 1e-12,
                )
                advantage = max(ratio - 1.0, 0.0)
                value = np.tanh(advantage)
            else:
                value = 1.0

            detour[i, j] = value
            detour[j, i] = value

    cycle_support = 0.20 + 0.80 * detour

    tree_edges = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue

        best_cost = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best_cost[start] = 0.0

        while True:
            candidates = np.where(~visited, best_cost, np.inf)
            u = int(np.argmin(candidates))

            if not np.isfinite(candidates[u]):
                break

            visited[u] = True

            if parent[u] >= 0:
                v = parent[u]
                tree_edges[u, v] = True
                tree_edges[v, u] = True

            neighbors = np.where(valid[u] & ~visited)[0]
            for v in neighbors:
                if distances[u, v] < best_cost[v]:
                    best_cost[v] = distances[u, v]
                    parent[v] = u

    connectivity = np.full((n, n), 0.55, dtype=float)
    connectivity[tree_edges] = 1.35
    connectivity[~valid] = 0.0

    edge_values_sorted = np.sort(values)
    percentile = (
        np.searchsorted(edge_values_sorted, values, side="right")
        / max(float(values.size), 1.0)
    )

    global_preference = np.zeros((n, n), dtype=float)
    global_preference[valid] = 0.08 + 0.92 * (1.0 - percentile)

    median_distance = float(np.median(values))
    mad = float(np.median(np.abs(values - median_distance)))
    temperature = max(mad, safe_scale * 0.05, eps)

    logistic = np.zeros((n, n), dtype=float)
    centered = (median_distance - distances[valid]) / temperature
    logistic[valid] = 0.15 + 0.85 / (
        1.0 + np.exp(-np.clip(centered, -40.0, 40.0))
    )

    features = [
        np.maximum(inverse, eps),
        np.maximum(mutual_competition, eps),
        np.maximum(mutual_rank, eps),
        np.maximum(contrast, eps),
        np.maximum(global_preference, eps),
        np.maximum(logistic, eps),
        np.maximum(cycle_support, eps),
        np.maximum(connectivity, eps),
    ]

    weights = np.array(
        [0.24, 0.16, 0.14, 0.12, 0.10, 0.08, 0.09, 0.07],
        dtype=float,
    )

    score = np.ones((n, n), dtype=float)
    for feature, weight in zip(features, weights):
        score *= np.power(feature, weight)

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    valid_scores = score[valid]
    floor = 0.04 * float(np.mean(valid_scores))
    score[valid] = 0.96 * score[valid] + floor

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
