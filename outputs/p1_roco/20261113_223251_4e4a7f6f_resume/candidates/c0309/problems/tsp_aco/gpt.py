import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return direction-aware, diverse edge priors for stochastic TSP sampling."""
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
        q25, q50, q75 = np.quantile(positive, [0.25, 0.50, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    scale = max(scale, eps)
    upper = max(
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.75)),
        scale,
    )
    replacement = upper + 18.0 * scale

    distances = np.where(valid, raw, replacement)
    np.fill_diagonal(distances, replacement)

    normalized = np.minimum(distances / scale, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(distances[valid], eps)
    inverse_max = float(np.max(inverse))
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        inverse /= inverse_max

    row_mass = affinity.sum(axis=1, keepdims=True)
    col_mass = affinity.sum(axis=0, keepdims=True)

    row_prob = affinity / np.maximum(row_mass, eps)
    col_prob = affinity / np.maximum(col_mass, eps)
    row_prob[~valid] = 0.0
    col_prob[~valid] = 0.0

    # Stable row and column competition signals.
    row_order = np.argsort(distances, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    col_order = np.argsort(distances, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    rank_temperature = max(1.0, 0.65 * np.sqrt(float(n)))
    rank_signal = 0.5 * (
        np.exp(-row_rank / rank_temperature)
        + np.exp(-col_rank / rank_temperature)
    )
    rank_signal[~valid] = 0.0

    # Robust global quality.
    sorted_values = np.sort(values)
    percentile = (
        np.searchsorted(sorted_values, values, side="right").astype(float)
        / max(float(values.size), 1.0)
    )

    global_quality = np.zeros((n, n), dtype=float)
    global_quality[valid] = 1.0 - percentile

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    temperature = max(mad, 0.18 * scale, eps)

    centered = (median - distances) / temperature
    smooth_distance = np.zeros((n, n), dtype=float)
    smooth_distance[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -40.0, 40.0))
    )
    smooth_distance[valid] = 0.20 + 0.80 * smooth_distance[valid]

    # Endpoint regret: preserve edges with few competitive alternatives.
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

    # Directed two-hop detour signal.
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
        -detour / max(0.90 * scale, eps)
    )
    detour_signal[~valid] = 0.0

    # Undirected connectivity skeleton used only as a mild bridge prior.
    undirected_cost = np.full((n, n), np.inf, dtype=float)
    both = valid & valid.T
    undirected_cost[both] = 0.5 * (
        raw[both] + raw.T[both]
    )

    only_forward = valid & ~valid.T
    undirected_cost[only_forward] = raw[only_forward]

    only_reverse = valid.T & ~valid
    undirected_cost[only_reverse] = raw.T[only_reverse]

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
            choices = np.where(~visited, best_edge, np.inf)
            node = int(np.argmin(choices))

            if not np.isfinite(choices[node]):
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

    bridge_signal = np.full((n, n), 0.12, dtype=float)
    bridge_signal[forest & valid] = 1.0
    bridge_signal[~valid] = 0.0

    # Directional scarcity protects rare successors and predecessors.
    successor_count = valid.sum(axis=1).astype(float)
    predecessor_count = valid.sum(axis=0).astype(float)

    successor_scarcity = np.zeros(n, dtype=float)
    predecessor_scarcity = np.zeros(n, dtype=float)

    active_successors = successor_count > 0.0
    active_predecessors = predecessor_count > 0.0

    successor_scarcity[active_successors] = 1.0 / np.sqrt(
        successor_count[active_successors]
    )
    predecessor_scarcity[active_predecessors] = 1.0 / np.sqrt(
        predecessor_count[active_predecessors]
    )

    if np.any(active_successors):
        successor_scarcity /= max(
            float(np.max(successor_scarcity)),
            eps,
        )

    if np.any(active_predecessors):
        predecessor_scarcity /= max(
            float(np.max(predecessor_scarcity)),
            eps,
        )

    scarcity = 0.5 * (
        successor_scarcity[:, None]
        + predecessor_scarcity[None, :]
    )
    scarcity[~valid] = 0.0

    # Base score: distance remains dominant and other terms are additive.
    base = (
        0.42 * smooth_distance
        + 0.18 * inverse
        + 0.16 * rank_signal
        + 0.14 * global_quality
        + 0.10 * affinity
    )

    long_term = (
        0.16 * regret
        + 0.12 * detour_signal
        + 0.08 * bridge_signal
        + 0.08 * scarcity
    )

    score = base + long_term
    score[~valid] = 0.0

    # Mild marginal novelty. It does not eliminate high-probability edges.
    row_novelty = 1.0 / (1.0 + 1.6 * row_prob)
    col_novelty = 1.0 / (1.0 + 1.6 * col_prob)
    novelty = np.sqrt(
        np.maximum(row_novelty * col_novelty, 0.0)
    )
    novelty[~valid] = 0.0

    score *= 0.90 + 0.10 * novelty
    score *= 0.96 + 0.04 * scarcity
    score[~valid] = 0.0

    positive = score[valid]
    if positive.size:
        floor = max(
            0.035 * float(np.max(positive)),
            eps,
        )
        score[valid] = np.maximum(score[valid], floor)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result[valid] = 1.0
        return result

    score /= maximum

    # Entropic coverage balancing with deliberately small exponent.
    # This redistributes mass without replacing the distance model.
    balanced = score.copy()
    active_rows = valid.any(axis=1)
    active_cols = valid.any(axis=0)

    for _ in range(3):
        row_sum = balanced.sum(axis=1)
        row_scale = np.ones(n, dtype=float)
        row_scale[active_rows] = np.power(
            1.0 / np.maximum(row_sum[active_rows], eps),
            0.08,
        )
        balanced *= row_scale[:, None]
        balanced[~valid] = 0.0

        col_sum = balanced.sum(axis=0)
        col_scale = np.ones(n, dtype=float)
        col_scale[active_cols] = np.power(
            1.0 / np.maximum(col_sum[active_cols], eps),
            0.08,
        )
        balanced *= col_scale[None, :]
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.84 * score + 0.16 * balanced
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
    ).astype(float, copy=False)
