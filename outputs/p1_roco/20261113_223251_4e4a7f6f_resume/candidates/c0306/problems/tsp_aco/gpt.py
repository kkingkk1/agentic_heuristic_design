import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return direction-aware, globally diverse edge priors for stochastic TSP sampling."""
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
    replacement = upper + 16.0 * scale

    distances = np.where(valid, raw, replacement)
    np.fill_diagonal(distances, replacement)

    clipped = np.minimum(distances / scale, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-clipped[valid])

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

    mutual = np.sqrt(np.maximum(row_prob * col_prob, 0.0))

    order_row = np.argsort(distances, axis=1, kind="stable")
    rank_row = np.empty((n, n), dtype=float)
    rank_row[np.arange(n)[:, None], order_row] = np.arange(n, dtype=float)

    order_col = np.argsort(distances, axis=0, kind="stable")
    rank_col = np.empty((n, n), dtype=float)
    rank_col[order_col, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    rank_temperature = max(1.0, 0.55 * np.sqrt(float(n)))
    rank_signal = 0.5 * (
        np.exp(-rank_row / rank_temperature)
        + np.exp(-rank_col / rank_temperature)
    )
    rank_signal[~valid] = 0.0

    finite_rows = np.where(valid, distances, np.inf)
    sorted_rows = np.sort(finite_rows, axis=1)

    best_row = sorted_rows[:, 0]
    second_row = sorted_rows[:, min(1, n - 1)]
    second_row = np.where(np.isfinite(second_row), second_row, best_row + scale)
    row_gap = np.maximum(second_row - best_row, 0.12 * scale)

    row_regret = np.maximum(second_row[:, None] - distances, 0.0)
    row_regret = 1.0 - np.exp(
        -row_regret / np.maximum(row_gap[:, None], eps)
    )

    finite_cols = np.where(valid, distances, np.inf)
    sorted_cols = np.sort(finite_cols, axis=0)

    best_col = sorted_cols[0, :]
    second_col = sorted_cols[min(1, n - 1), :]
    second_col = np.where(np.isfinite(second_col), second_col, best_col + scale)
    col_gap = np.maximum(second_col - best_col, 0.12 * scale)

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
        -detour / max(0.80 * scale, eps)
    )
    detour_signal[~valid] = 0.0

    undirected_cost = np.full((n, n), np.inf, dtype=float)
    reverse_valid = valid.T

    both = valid & reverse_valid
    undirected_cost[both] = 0.5 * (raw[both] + raw.T[both])
    only_forward = valid & ~reverse_valid
    undirected_cost[only_forward] = raw[only_forward]
    only_reverse = reverse_valid & ~valid
    undirected_cost[only_reverse] = raw.T[only_reverse]

    np.fill_diagonal(undirected_cost, np.inf)

    forest = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue

        component_root = start
        best_edge = np.full(n, np.inf, dtype=float)
        best_edge[component_root] = 0.0
        parent = np.full(n, -1, dtype=int)

        while True:
            candidate_values = np.where(~visited, best_edge, np.inf)
            node = int(np.argmin(candidate_values))

            if not np.isfinite(candidate_values[node]):
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

    bridge_signal = np.full((n, n), 0.16, dtype=float)
    bridge_signal[forest & valid] = 1.0

    one_way_bridge = forest & valid & ~valid.T
    bridge_signal[one_way_bridge] = 0.88
    bridge_signal[~valid] = 0.0

    sorted_values = np.sort(values)
    positions = np.searchsorted(sorted_values, values, side="right")
    percentile = positions.astype(float) / max(float(values.size), 1.0)

    global_quality = np.zeros((n, n), dtype=float)
    global_quality[valid] = 1.0 - percentile

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    temperature = max(mad, 0.12 * scale, eps)

    centered = (median - distances) / temperature
    smooth_distance = np.zeros((n, n), dtype=float)
    smooth_distance[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -40.0, 40.0))
    )
    smooth_distance[valid] = 0.18 + 0.82 * smooth_distance[valid]

    row_entropy = np.zeros(n, dtype=float)
    col_entropy = np.zeros(n, dtype=float)

    for node in range(n):
        probabilities = row_prob[node, valid[node]]
        if probabilities.size > 1:
            entropy = -np.sum(
                probabilities * np.log(np.maximum(probabilities, eps))
            )
            row_entropy[node] = entropy / max(
                np.log(float(probabilities.size)),
                1.0,
            )
        elif probabilities.size == 1:
            row_entropy[node] = 0.0

        probabilities = col_prob[valid[:, node], node]
        if probabilities.size > 1:
            entropy = -np.sum(
                probabilities * np.log(np.maximum(probabilities, eps))
            )
            col_entropy[node] = entropy / max(
                np.log(float(probabilities.size)),
                1.0,
            )
        elif probabilities.size == 1:
            col_entropy[node] = 0.0

    endpoint_entropy = np.sqrt(
        np.maximum(
            row_entropy[:, None] * col_entropy[None, :],
            0.0,
        )
    )

    row_novelty = 1.0 / (1.0 + 2.2 * row_prob)
    col_novelty = 1.0 / (1.0 + 2.2 * col_prob)
    novelty = np.sqrt(np.maximum(row_novelty * col_novelty, 0.0))
    novelty[~valid] = 0.0

    base = (
        0.28 * smooth_distance
        + 0.16 * inverse
        + 0.12 * mutual
        + 0.12 * rank_signal
        + 0.12 * global_quality
    )

    long_term = (
        0.17 * regret
        + 0.13 * detour_signal
        + 0.10 * bridge_signal
    )

    score = base + long_term

    score *= 0.76 + 0.24 * endpoint_entropy
    score *= 0.72 + 0.28 * novelty
    score[~valid] = 0.0

    positive_scores = score[valid]
    floor = max(
        0.025 * float(np.max(positive_scores)),
        eps,
    )
    score[valid] = np.maximum(score[valid], floor)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        score = valid.astype(float)
        np.fill_diagonal(score, 0.0)
    else:
        score /= maximum

    balanced = score.copy()
    active_rows = valid.any(axis=1)
    active_cols = valid.any(axis=0)

    for _ in range(4):
        row_sum = balanced.sum(axis=1)
        row_scale = np.ones(n, dtype=float)
        row_scale[active_rows] = np.power(
            1.0 / np.maximum(row_sum[active_rows], eps),
            0.16,
        )
        balanced *= row_scale[:, None]
        balanced[~valid] = 0.0

        col_sum = balanced.sum(axis=0)
        col_scale = np.ones(n, dtype=float)
        col_scale[active_cols] = np.power(
            1.0 / np.maximum(col_sum[active_cols], eps),
            0.16,
        )
        balanced *= col_scale[None, :]
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.78 * score + 0.22 * balanced
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
