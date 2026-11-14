import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust direction-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    values = raw[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        q25, q75 = np.quantile(positive, [0.25, 0.75])
        mad = float(np.median(np.abs(positive - median)))
        scale = max(
            float(q75 - q25) / 1.349,
            mad / 0.6745,
            0.10 * max(median, 1.0),
            eps,
        )
    else:
        median = 0.0
        scale = 1.0

    maximum = float(np.max(values))
    replacement = maximum + 8.0 * scale
    distances = np.where(valid, raw, replacement)
    np.fill_diagonal(distances, replacement)

    scaled = np.clip(distances / max(scale, eps), 0.0, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-scaled[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(distances[valid], eps)
    inverse_max = float(np.max(inverse))
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        inverse /= inverse_max

    distance_signal = 0.72 * affinity + 0.28 * inverse
    distance_signal[~valid] = 0.0

    row_mass = distance_signal.sum(axis=1, keepdims=True)
    col_mass = distance_signal.sum(axis=0, keepdims=True)

    row_preference = distance_signal / np.maximum(row_mass, eps)
    col_preference = distance_signal / np.maximum(col_mass, eps)

    mutual = np.sqrt(np.maximum(row_preference * col_preference, 0.0))
    mutual[~valid] = 0.0

    order_row = np.argsort(distances, axis=1, kind="stable")
    rank_row = np.empty((n, n), dtype=float)
    rank_row[np.arange(n)[:, None], order_row] = np.arange(n, dtype=float)

    order_col = np.argsort(distances, axis=0, kind="stable")
    rank_col = np.empty((n, n), dtype=float)
    rank_col[order_col, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    rank_temperature = max(1.0, 0.65 * np.sqrt(float(n)))
    rank_signal = 0.5 * (
        np.exp(-rank_row / rank_temperature)
        + np.exp(-rank_col / rank_temperature)
    )
    rank_signal[~valid] = 0.0

    finite_rows = np.where(valid, distances, np.inf)
    sorted_rows = np.sort(finite_rows, axis=1)

    best_row = sorted_rows[:, 0]
    second_row = sorted_rows[:, min(1, n - 1)]
    second_row = np.where(
        np.isfinite(second_row),
        second_row,
        best_row + scale,
    )

    row_gap = np.maximum(second_row - best_row, 0.15 * scale)
    row_regret = np.maximum(second_row[:, None] - distances, 0.0)
    row_regret = 1.0 - np.exp(
        -row_regret / np.maximum(row_gap[:, None], eps)
    )

    finite_cols = np.where(valid, distances, np.inf)
    sorted_cols = np.sort(finite_cols, axis=0)

    best_col = sorted_cols[0, :]
    second_col = sorted_cols[min(1, n - 1), :]
    second_col = np.where(
        np.isfinite(second_col),
        second_col,
        best_col + scale,
    )

    col_gap = np.maximum(second_col - best_col, 0.15 * scale)
    col_regret = np.maximum(second_col[None, :] - distances, 0.0)
    col_regret = 1.0 - np.exp(
        -col_regret / np.maximum(col_gap[None, :], eps)
    )

    regret = 0.5 * (row_regret + col_regret)
    regret[~valid] = 0.0

    bypass = np.full((n, n), replacement, dtype=float)
    for middle in range(n):
        candidate = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = replacement
    detour = np.maximum(bypass - distances, 0.0)
    detour_signal = 1.0 - np.exp(
        -detour / max(0.8 * scale, eps)
    )
    detour_signal[~valid] = 0.0

    global_quality = np.zeros((n, n), dtype=float)
    sorted_values = np.sort(values)
    percentile = (
        np.searchsorted(sorted_values, values, side="right")
        / max(float(values.size), 1.0)
    )
    global_quality[valid] = 1.0 - percentile

    centered = (median - distances) / max(0.75 * scale, eps)
    smooth_distance = np.zeros((n, n), dtype=float)
    smooth_distance[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -40.0, 40.0))
    )
    smooth_distance[valid] = 0.18 + 0.82 * smooth_distance[valid]

    undirected_cost = np.full((n, n), np.inf, dtype=float)
    reverse_valid = valid.T
    both = valid & reverse_valid
    forward_only = valid & ~reverse_valid
    reverse_only = reverse_valid & ~valid

    undirected_cost[both] = 0.5 * (raw[both] + raw.T[both])
    undirected_cost[forward_only] = raw[forward_only]
    undirected_cost[reverse_only] = raw.T[reverse_only]
    np.fill_diagonal(undirected_cost, np.inf)

    forest = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue

        best_edge = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best_edge[start] = 0.0

        while True:
            candidates = np.where(~visited, best_edge, np.inf)
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            visited[node] = True
            predecessor = parent[node]

            if predecessor >= 0:
                forest[node, predecessor] = True
                forest[predecessor, node] = True

            neighbors = np.flatnonzero(
                np.isfinite(undirected_cost[node]) & ~visited
            )

            for other in neighbors:
                edge_cost = undirected_cost[node, other]
                if edge_cost < best_edge[other]:
                    best_edge[other] = edge_cost
                    parent[other] = node

    bridge_signal = np.full((n, n), 0.18, dtype=float)
    bridge_signal[forest & valid] = 1.0
    bridge_signal[~valid] = 0.0

    score = (
        0.34 * smooth_distance
        + 0.16 * distance_signal
        + 0.14 * mutual
        + 0.10 * rank_signal
        + 0.10 * regret
        + 0.08 * detour_signal
        + 0.05 * bridge_signal
        + 0.03 * global_quality
    )
    score[~valid] = 0.0

    positive = score[valid]
    if positive.size:
        maximum = float(np.max(positive))
        floor = max(0.025 * maximum, eps)
        score[valid] = floor + 0.975 * score[valid]

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result[valid] = 1.0
        np.fill_diagonal(result, 0.0)
        return result

    result = score / maximum
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
