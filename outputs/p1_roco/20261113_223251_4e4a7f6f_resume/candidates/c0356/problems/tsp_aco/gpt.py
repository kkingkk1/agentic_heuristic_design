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

    finite_max = float(np.max(edge_values))
    penalty = finite_max + 18.0 * safe_scale

    distances = np.where(
        valid,
        symmetric,
        penalty,
    )
    np.fill_diagonal(distances, penalty)

    normalized = np.minimum(
        distances / safe_scale,
        50.0,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-0.82 * normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        distances[valid] + safe_scale * 1e-9
    )
    inverse_max = float(np.max(inverse))
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        inverse /= inverse_max

    row_mass = affinity.sum(axis=1)
    col_mass = affinity.sum(axis=0)

    row_preference = affinity / np.maximum(
        row_mass[:, None],
        eps,
    )
    col_preference = affinity / np.maximum(
        col_mass[None, :],
        eps,
    )

    mutual_preference = np.sqrt(
        np.maximum(
            row_preference * col_preference,
            0.0,
        )
    )
    mutual_preference[~valid] = 0.0

    order = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(
        1.0,
        0.95 * np.sqrt(float(n)),
    )
    rank_preference = np.exp(
        -ranks / rank_scale
    )
    rank_preference[~valid] = 0.0

    mutual_max = max(
        float(np.max(mutual_preference)),
        eps,
    )
    local_rank = (
        0.48 * rank_preference
        + 0.52 * mutual_preference / mutual_max
    )
    local_rank[~valid] = 0.0

    row_sorted = np.sort(distances, axis=1)
    best = row_sorted[:, 0]
    second = row_sorted[:, min(1, n - 1)]

    gap_scale = np.maximum(
        second - best,
        0.16 * safe_scale,
    )

    row_gap = np.maximum(
        second[:, None] - distances,
        0.0,
    )
    row_exclusivity = 1.0 - np.exp(
        -row_gap / gap_scale[:, None]
    )
    row_exclusivity[~valid] = 0.0

    col_sorted = np.sort(distances, axis=0)
    col_best = col_sorted[0, :]
    col_second = col_sorted[min(1, n - 1), :]

    col_gap_scale = np.maximum(
        col_second - col_best,
        0.16 * safe_scale,
    )

    col_gap = np.maximum(
        col_second[None, :] - distances,
        0.0,
    )
    col_exclusivity = 1.0 - np.exp(
        -col_gap / col_gap_scale[None, :]
    )
    col_exclusivity[~valid] = 0.0

    exclusivity = np.sqrt(
        np.maximum(
            row_exclusivity * col_exclusivity,
            0.0,
        )
    )
    exclusivity[~valid] = 0.0

    best_bypass = np.full(
        (n, n),
        penalty,
        dtype=float,
    )

    for middle in range(n):
        candidate = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        best_bypass = np.minimum(
            best_bypass,
            candidate,
        )

    np.fill_diagonal(best_bypass, penalty)

    bypass_advantage = np.maximum(
        best_bypass - distances,
        0.0,
    )
    bypass_scale = max(
        0.75 * safe_scale,
        eps,
    )
    irreplaceability = 1.0 - np.exp(
        -bypass_advantage / bypass_scale
    )
    irreplaceability[~valid] = 0.0

    structural = (
        0.48 * exclusivity
        + 0.52 * irreplaceability
    )
    structural[~valid] = 0.0

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
        0.90,
        dtype=float,
    )
    connectivity[forest_edges] = 1.13
    connectivity[~valid] = 0.0

    ordered_values = np.sort(edge_values)
    edge_percentiles = (
        np.searchsorted(
            ordered_values,
            edge_values,
            side="right",
        ).astype(float)
        / max(float(edge_values.size), 1.0)
    )

    global_rank = np.zeros((n, n), dtype=float)
    global_rank[valid] = (
        0.25 + 0.75 * (1.0 - edge_percentiles)
    )

    median_distance = float(np.median(edge_values))
    mad = float(
        np.median(
            np.abs(edge_values - median_distance)
        )
    )
    temperature = max(
        mad,
        0.10 * safe_scale,
        eps,
    )

    centered = (
        median_distance - distances
    ) / temperature

    smooth_distance = np.zeros(
        (n, n),
        dtype=float,
    )
    smooth_distance[valid] = (
        0.20
        + 0.80 / (
            1.0 + np.exp(
                -np.clip(
                    centered[valid],
                    -35.0,
                    35.0,
                )
            )
        )
    )

    base = (
        np.maximum(global_rank, eps) ** 0.70
        * np.maximum(smooth_distance, eps) ** 0.92
        * np.maximum(inverse, eps) ** 0.72
        * np.maximum(local_rank, eps) ** 1.18
        * np.maximum(structural + 0.18, eps) ** 1.02
        * np.maximum(connectivity, eps)
    )

    base[~valid] = 0.0
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    maximum = float(np.max(base))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    base /= maximum

    balanced = base.copy()

    for _ in range(24):
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

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = (
        0.72 * base
        + 0.28 * balanced
    )

    result[~valid] = 0.0
    result = 0.5 * (
        result + result.T
    )
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
