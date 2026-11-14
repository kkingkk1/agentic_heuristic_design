import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

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

    edge_values = symmetric[valid]
    positive = edge_values[edge_values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    robust_upper = max(
        float(np.quantile(edge_values, 0.93)),
        float(np.quantile(edge_values, 0.72)),
        safe_scale,
    )
    replacement = robust_upper + 10.0 * safe_scale

    distances = np.where(
        valid,
        symmetric,
        replacement,
    )
    np.fill_diagonal(distances, replacement)

    normalized = np.minimum(
        distances / safe_scale,
        60.0,
    )

    distance_affinity = np.zeros((n, n), dtype=float)
    distance_affinity[valid] = np.exp(
        -normalized[valid]
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        distances[valid] + safe_scale * 1e-9
    )
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = distance_affinity.sum(axis=1)
    col_mass = distance_affinity.sum(axis=0)

    row_preference = (
        distance_affinity
        / np.maximum(row_mass[:, None], eps)
    )
    col_preference = (
        distance_affinity
        / np.maximum(col_mass[None, :], eps)
    )

    mutual_preference = np.sqrt(
        np.maximum(
            row_preference * col_preference,
            eps,
        )
    )
    mutual_preference[~valid] = 0.0

    row_order = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_order = np.argsort(
        distances,
        axis=0,
        kind="stable",
    )
    col_rank = np.empty((n, n), dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)

    rank_scale = max(
        1.0,
        0.62 * np.sqrt(float(n)),
    )

    row_rank_score = np.exp(
        -row_rank / rank_scale
    )
    col_rank_score = np.exp(
        -col_rank / rank_scale
    )

    mutual_rank = np.sqrt(
        np.maximum(
            row_rank_score * col_rank_score,
            eps,
        )
    )
    mutual_rank[~valid] = 0.0

    row_sorted = np.sort(distances, axis=1)
    row_best = row_sorted[:, 0]
    row_second = row_sorted[:, min(1, n - 1)]
    row_gap_scale = np.maximum(
        row_second - row_best,
        0.10 * safe_scale,
    )

    col_sorted = np.sort(distances, axis=0)
    col_best = col_sorted[0, :]
    col_second = col_sorted[min(1, n - 1), :]
    col_gap_scale = np.maximum(
        col_second - col_best,
        0.10 * safe_scale,
    )

    row_gap = np.maximum(
        row_second[:, None] - distances,
        0.0,
    )
    col_gap = np.maximum(
        col_second[None, :] - distances,
        0.0,
    )

    row_exclusivity = 1.0 - np.exp(
        -row_gap / row_gap_scale[:, None]
    )
    col_exclusivity = 1.0 - np.exp(
        -col_gap / col_gap_scale[None, :]
    )

    exclusivity = np.sqrt(
        np.maximum(
            row_exclusivity * col_exclusivity,
            0.0,
        )
    )
    exclusivity[~valid] = 0.0

    bypass = np.full(
        (n, n),
        replacement,
        dtype=float,
    )

    for intermediate in range(n):
        candidate = (
            distances[:, intermediate, None]
            + distances[None, intermediate, :]
        )
        bypass = np.minimum(
            bypass,
            candidate,
        )

    bypass[diagonal] = replacement

    detour_advantage = np.maximum(
        bypass - distances,
        0.0,
    )

    detour_scale = max(
        0.52 * safe_scale,
        eps,
    )
    irreplaceability = 1.0 - np.exp(
        -detour_advantage / detour_scale
    )
    irreplaceability[~valid] = 0.0

    forest_edges = np.zeros(
        (n, n),
        dtype=bool,
    )
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

    connectivity = np.full(
        (n, n),
        0.88,
        dtype=float,
    )
    connectivity[forest_edges] = 1.16
    connectivity[~valid] = 0.0

    sorted_values = np.sort(edge_values)
    percentile = (
        np.searchsorted(
            sorted_values,
            edge_values,
            side="right",
        ).astype(float)
        / max(float(edge_values.size), 1.0)
    )

    global_rank = np.zeros((n, n), dtype=float)
    global_rank[valid] = 1.0 - percentile
    global_rank[valid] = 0.16 + 0.84 * global_rank[valid]

    median_distance = float(np.median(edge_values))
    mad = float(
        np.median(
            np.abs(edge_values - median_distance)
        )
    )
    temperature = max(
        mad,
        0.08 * safe_scale,
        eps,
    )

    centered = (
        median_distance - distances
    ) / temperature

    smooth_distance = np.zeros(
        (n, n),
        dtype=float,
    )
    smooth_distance[valid] = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                centered[valid],
                -40.0,
                40.0,
            )
        )
    )
    smooth_distance[valid] = (
        0.15
        + 0.85 * smooth_distance[valid]
    )

    local_signal = (
        0.50 * mutual_rank
        + 0.50 * mutual_preference
    )
    local_signal[~valid] = 0.0

    structural_signal = (
        0.64 * irreplaceability
        + 0.36 * exclusivity
    )
    structural_signal[~valid] = 0.0

    score = (
        np.maximum(global_rank, eps)
        * np.maximum(smooth_distance, eps)
        * np.maximum(inverse, eps)
        * np.maximum(local_signal, eps)
        * np.maximum(structural_signal + 0.16, eps)
        * np.maximum(connectivity, eps)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        score = valid.astype(float)
        np.fill_diagonal(score, 0.0)
    else:
        score /= maximum

    balanced = score.copy()

    for _ in range(18):
        row_sum = balanced.sum(
            axis=1,
            keepdims=True,
        )
        balanced /= np.maximum(
            row_sum,
            eps,
        )
        balanced = 0.5 * (
            balanced + balanced.T
        )
        balanced[~valid] = 0.0

    result = (
        0.70 * score
        + 0.30 * balanced
    )

    result[~valid] = 0.0
    result = 0.5 * (
        result + result.T
    )
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
