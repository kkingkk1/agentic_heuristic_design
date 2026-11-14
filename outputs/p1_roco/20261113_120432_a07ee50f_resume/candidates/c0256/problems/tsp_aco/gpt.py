import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid_input = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]
    row_rank = np.where(valid, row_rank, np.inf)

    column_order = np.argsort(safe, axis=0, kind="stable")
    column_rank = np.full((n, n), float(n), dtype=float)
    column_rank[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]
    column_rank = np.where(valid, column_rank, np.inf)

    # Local scale from the nearest few usable neighbors.
    local_k = min(3, n - 1)
    nearest_distances = np.take_along_axis(
        safe,
        order[:, :local_k],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_distances)
    local_counts = nearest_valid.sum(axis=1)

    local_scale = (
        np.sum(
            np.where(nearest_valid, nearest_distances, 0.0),
            axis=1,
        )
        / np.maximum(local_counts, 1)
    )
    local_scale = np.where(
        local_counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    normalized_distance = safe / pair_scale
    distance_log = -np.clip(normalized_distance, 0.0, 50.0)

    rank_log = -0.5 * (
        np.log1p(row_rank) + np.log1p(column_rank)
    )

    endpoint_affinity = -0.5 * (
        np.clip(safe / local_scale[:, None], 0.0, 50.0)
        + np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )

    # Conservative opportunity-cost penalty.
    alternative_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    regret = np.maximum(
        safe / np.maximum(alternative_scale, eps) - 1.0,
        0.0,
    )
    regret_penalty = -0.14 * np.clip(regret, 0.0, 4.0)

    # Mutual-neighborhood rewards from the exploiter.
    top_k = min(2, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True

    mutual_top = (
        valid
        & top_neighbors
        & top_neighbors.T
    ).astype(float)

    nearest_row = order[:, 0]
    nearest_column = column_order[0, :]
    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_column[None, :] == np.arange(n)[:, None])
    ).astype(float)

    mutual_signal = (
        0.11 * mutual_top
        + 0.16 * mutual_nearest
    )

    # Mild global calibration for heterogeneous distance scales.
    global_quality_log = -np.clip(safe / global_scale, 0.0, 50.0)
    scale_ratio = np.log(
        np.maximum(local_scale / global_scale, eps)
    )
    heterogeneity = float(
        np.clip(np.std(scale_ratio) / 1.5, 0.0, 1.0)
    )
    global_weight = 0.06 + 0.04 * heterogeneity
    global_calibration = global_weight * (
        global_quality_log - distance_log
    )

    # Multi-scale neighborhood disagreement gives a bounded bridge bonus.
    small_k = min(max(2, int(np.sqrt(max(n, 2)))), n - 1)
    large_k = min(
        max(small_k + 2, int(0.35 * (n - 1))),
        n - 1,
    )

    small_mask = np.zeros((n, n), dtype=bool)
    large_mask = np.zeros((n, n), dtype=bool)

    small_mask[
        np.arange(n)[:, None],
        order[:, :small_k],
    ] = True
    large_mask[
        np.arange(n)[:, None],
        order[:, :large_k],
    ] = True

    small_mask &= valid
    large_mask &= valid

    small_degree = np.maximum(
        small_mask.sum(axis=1).astype(float),
        1.0,
    )
    large_degree = np.maximum(
        large_mask.sum(axis=1).astype(float),
        1.0,
    )

    small_overlap = (
        small_mask.astype(float) @ small_mask.astype(float).T
    ) / np.sqrt(
        small_degree[:, None] * small_degree[None, :]
    )
    large_overlap = (
        large_mask.astype(float) @ large_mask.astype(float).T
    ) / np.sqrt(
        large_degree[:, None] * large_degree[None, :]
    )

    small_overlap = np.clip(small_overlap, 0.0, 1.0)
    large_overlap = np.clip(large_overlap, 0.0, 1.0)

    disagreement = (
        0.60 * (1.0 - small_overlap)
        + 0.40 * (1.0 - large_overlap)
    )

    mutual_support = np.sqrt(
        np.clip(
            0.5 * (small_mask.astype(float) + large_mask.astype(float)),
            0.0,
            1.0,
        )
        * np.clip(
            0.5 * (
                small_mask.astype(float).T
                + large_mask.astype(float).T
            ),
            0.0,
            1.0,
        )
    )

    density = float(valid.sum()) / max(float(n * (n - 1)), 1.0)
    sparsity = 1.0 - density

    bridge_log = (
        (0.025 + 0.035 * sparsity)
        * disagreement
        * (0.5 + 0.5 * mutual_support)
    )

    redundancy_log = (
        (0.015 + 0.015 * (1.0 - sparsity))
        * (
            0.65 * small_overlap
            + 0.35 * large_overlap
        )
        * mutual_support
    )

    diversity_log = bridge_log - redundancy_log

    # Light endpoint-fragility protection from the explorer.
    sorted_values = np.sort(
        np.where(valid, safe, np.inf),
        axis=1,
    )
    nearest = sorted_values[:, 0]
    second = sorted_values[:, min(1, n - 2)]

    regret_by_node = np.maximum(second - nearest, 0.0)
    regret_by_node = regret_by_node / np.maximum(local_scale, eps)
    regret_by_node = np.clip(regret_by_node, 0.0, 6.0)

    degree = valid.sum(axis=1).astype(float)
    positive_degree = degree[degree > 0.0]
    degree_reference = (
        max(float(np.median(positive_degree)), 1.0)
        if positive_degree.size
        else 1.0
    )

    scarcity = np.clip(
        degree_reference / np.maximum(degree, 1.0),
        0.5,
        2.5,
    )

    flexibility = np.clip(
        large_degree / max(float(large_k), 1.0),
        0.0,
        1.0,
    )

    fragility = np.clip(
        0.58 * (regret_by_node / 3.0)
        + 0.42 * ((scarcity - 0.5) / 2.0),
        0.0,
        1.0,
    )

    endpoint_fragility = np.sqrt(
        (1.0 + fragility[:, None])
        * (1.0 + fragility[None, :])
    ) - 1.0

    shared_lock = np.sqrt(
        np.clip(
            (1.0 - flexibility[:, None])
            * (1.0 - flexibility[None, :]),
            0.0,
            1.0,
        )
    )

    long_term_signal = np.clip(
        0.62 * endpoint_fragility
        + 0.28 * shared_lock
        - 0.18 * (
            1.0
            - np.minimum(
                fragility[:, None],
                fragility[None, :],
            )
        ),
        -0.5,
        1.5,
    )

    long_term_weight = 0.035 + 0.035 * (
        0.55 * sparsity
        + 0.45 * heterogeneity
    )
    long_term_log = np.clip(
        long_term_weight * long_term_signal,
        -0.08,
        0.14,
    )

    log_scores = (
        0.42 * distance_log
        + 0.22 * rank_log
        + 0.22 * endpoint_affinity
        + regret_penalty
        + mutual_signal
        + global_calibration
        + diversity_log
        + long_term_log
    )

    log_scores[~valid] = -np.inf

    finite_scores = np.where(
        np.isfinite(log_scores),
        log_scores,
        0.0,
    )

    valid_max = float(np.max(finite_scores[valid]))
    scores = np.zeros((n, n), dtype=float)
    scores[valid] = np.exp(
        np.clip(
            finite_scores[valid] - valid_max,
            -50.0,
            0.0,
        )
    )

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
