import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, diversity-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return result

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(symmetric, np.inf)

    values = symmetric[valid]
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

    eps = np.finfo(float).eps
    scale = max(scale, eps)

    upper = max(
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.75)),
        scale,
    )
    replacement = upper + 16.0 * scale

    distances = np.where(valid, symmetric, replacement)
    np.fill_diagonal(distances, replacement)

    clipped = np.minimum(distances / scale, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-clipped[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(distances[valid], eps)
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = affinity.sum(axis=1)
    col_mass = affinity.sum(axis=0)
    row_prob = affinity / np.maximum(row_mass[:, None], eps)
    col_prob = affinity / np.maximum(col_mass[None, :], eps)

    mutual = np.sqrt(np.maximum(row_prob * col_prob, 0.0))
    mutual[~valid] = 0.0

    order_row = np.argsort(distances, axis=1, kind="stable")
    rank_row = np.empty((n, n), dtype=float)
    rank_row[np.arange(n)[:, None], order_row] = np.arange(n, dtype=float)

    order_col = np.argsort(distances, axis=0, kind="stable")
    rank_col = np.empty((n, n), dtype=float)
    rank_col[order_col, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    rank_temperature = max(1.0, 0.60 * np.sqrt(float(n)))
    rank_signal = 0.5 * (
        np.exp(-rank_row / rank_temperature)
        + np.exp(-rank_col / rank_temperature)
    )
    rank_signal[~valid] = 0.0

    sorted_rows = np.sort(distances, axis=1)
    best = sorted_rows[:, 0]
    second = sorted_rows[:, min(1, n - 1)]
    regret_scale = np.maximum(second - best, 0.15 * scale)

    row_regret = np.maximum(second[:, None] - distances, 0.0)
    row_regret = 1.0 - np.exp(
        -row_regret / np.maximum(regret_scale[:, None], eps)
    )

    sorted_cols = np.sort(distances, axis=0)
    best_col = sorted_cols[0, :]
    second_col = sorted_cols[min(1, n - 1), :]
    regret_scale_col = np.maximum(second_col - best_col, 0.15 * scale)

    col_regret = np.maximum(second_col[None, :] - distances, 0.0)
    col_regret = 1.0 - np.exp(
        -col_regret / np.maximum(regret_scale_col[None, :], eps)
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
        -detour / max(0.75 * scale, eps)
    )
    detour_signal[~valid] = 0.0

    forest = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)
    best_edge = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for start in range(n):
        if visited[start]:
            continue

        best_edge[start] = 0.0

        while True:
            candidates = np.where(~visited, best_edge, np.inf)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                break

            visited[node] = True
            if parent[node] >= 0:
                other = parent[node]
                forest[node, other] = True
                forest[other, node] = True

            neighbors = np.flatnonzero(valid[node] & ~visited)
            for other in neighbors:
                if distances[node, other] < best_edge[other]:
                    best_edge[other] = distances[node, other]
                    parent[other] = node

    bridge_signal = np.full((n, n), 0.18, dtype=float)
    bridge_signal[forest] = 1.0
    bridge_signal[~valid] = 0.0

    finite_rank = np.zeros((n, n), dtype=float)
    sorted_values = np.sort(values)
    positions = np.searchsorted(sorted_values, values, side="right")
    percentile = positions.astype(float) / max(float(values.size), 1.0)
    global_quality = 1.0 - percentile

    finite_rank[valid] = 0.25 + 0.75 * global_quality

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    temperature = max(mad, 0.12 * scale, eps)

    centered = (median - distances) / temperature
    smooth_distance = np.zeros((n, n), dtype=float)
    smooth_distance[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -40.0, 40.0))
    )
    smooth_distance[valid] = 0.20 + 0.80 * smooth_distance[valid]

    endpoint_entropy = np.zeros(n, dtype=float)
    for node in range(n):
        probabilities = row_prob[node, valid[node]]
        if probabilities.size:
            entropy = -float(
                np.sum(probabilities * np.log(np.maximum(probabilities, eps)))
            )
            endpoint_entropy[node] = entropy / max(
                np.log(float(probabilities.size)),
                1.0,
            )

    diversity = np.sqrt(
        np.maximum(
            endpoint_entropy[:, None] * endpoint_entropy[None, :],
            0.0,
        )
    )

    concentration = np.maximum(row_prob, col_prob)
    crowding = 1.0 / (1.0 + 1.6 * concentration)
    crowding[~valid] = 0.0

    base = (
        0.24 * smooth_distance
        + 0.16 * inverse
        + 0.14 * mutual
        + 0.12 * rank_signal
        + 0.12 * finite_rank
    )

    long_term = (
        0.18 * regret
        + 0.12 * detour_signal
        + 0.08 * bridge_signal
    )

    score = base + long_term
    score *= 0.72 + 0.28 * diversity
    score *= 0.78 + 0.22 * crowding
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        score = valid.astype(float)
        np.fill_diagonal(score, 0.0)
    else:
        score /= maximum

    balanced = score.copy()
    for _ in range(4):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.78 * score + 0.22 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
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
