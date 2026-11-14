import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(symmetric, np.inf)

    finite_edges = symmetric[valid]
    positive_edges = finite_edges[finite_edges > 0.0]

    if positive_edges.size:
        scale = float(np.median(positive_edges))
        spread = float(np.median(np.abs(positive_edges - scale)))
        scale = max(scale, spread, np.finfo(float).eps)
    else:
        scale = 1.0

    eps = np.finfo(float).eps
    finite_max = float(np.max(finite_edges))
    penalty = finite_max + 32.0 * scale

    distances = np.where(valid, symmetric, penalty)
    np.fill_diagonal(distances, penalty)

    safe = np.maximum(distances, 0.0)
    normalized = np.minimum(safe / scale, 60.0)

    shortness = np.zeros((n, n), dtype=float)
    shortness[valid] = np.exp(-normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + 1e-9 * scale)
    inverse_max = float(np.max(inverse))
    if inverse_max > 0.0 and np.isfinite(inverse_max):
        inverse /= inverse_max

    row_order = np.argsort(distances, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_temperature = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_score = np.exp(-row_rank / rank_temperature)
    rank_score[~valid] = 0.0

    row_mass = shortness.sum(axis=1)
    preference = shortness / np.maximum(
        row_mass[:, None],
        eps,
    )
    mutual = np.sqrt(
        np.maximum(preference * preference.T, 0.0)
    )
    mutual[~valid] = 0.0

    sorted_rows = np.sort(distances, axis=1)
    best = sorted_rows[:, 0]
    second = sorted_rows[:, min(1, n - 1)]

    row_gap_scale = np.maximum(
        second - best,
        0.15 * scale,
    )
    row_regret = np.maximum(
        second[:, None] - distances,
        0.0,
    ) / row_gap_scale[:, None]
    row_regret = 1.0 - np.exp(-row_regret)
    row_regret[~valid] = 0.0

    scarcity = np.sqrt(
        np.maximum(row_regret * row_regret.T, 0.0)
    )
    scarcity[~valid] = 0.0

    best_bypass = np.full((n, n), penalty, dtype=float)
    for middle in range(n):
        two_hop = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        best_bypass = np.minimum(best_bypass, two_hop)

    np.fill_diagonal(best_bypass, penalty)

    bypass_gain = np.maximum(best_bypass - distances, 0.0)
    bypass_scale = max(0.75 * scale, eps)
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / bypass_scale
    )
    irreplaceability[~valid] = 0.0

    forest_edges = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)
    best_edge = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for start in range(n):
        if visited[start]:
            continue

        best_edge[start] = 0.0

        while True:
            candidates = np.where(
                ~visited,
                best_edge,
                np.inf,
            )
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            visited[node] = True

            if parent[node] >= 0:
                other = parent[node]
                forest_edges[node, other] = True
                forest_edges[other, node] = True

            neighbors = np.flatnonzero(
                valid[node] & ~visited
            )
            for other in neighbors:
                if distances[node, other] < best_edge[other]:
                    best_edge[other] = distances[node, other]
                    parent[other] = node

    connectivity = np.ones((n, n), dtype=float)
    connectivity[forest_edges] = 1.22
    connectivity[~valid] = 0.0

    structural = (
        0.42 * mutual
        + 0.34 * scarcity
        + 0.24 * irreplaceability
    )
    structural[~valid] = 0.0

    score = (
        np.maximum(shortness, eps) ** 1.25
        * np.maximum(inverse, eps) ** 0.55
        * np.maximum(rank_score, eps) ** 0.85
        * np.maximum(mutual + 0.08, eps) ** 0.70
        * np.maximum(structural + 0.12, eps) ** 0.95
        * connectivity
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    degree_mass = score.sum(axis=1)
    balancing = 1.0 / np.sqrt(
        np.maximum(degree_mass[:, None] * degree_mass[None, :], eps)
    )
    balanced = score * balancing
    balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.78 * score + 0.22 * balanced
    result[valid] = 0.96 * result[valid] + 0.04
    result[~valid] = 0.0

    result = 0.5 * (result + result.T)
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
