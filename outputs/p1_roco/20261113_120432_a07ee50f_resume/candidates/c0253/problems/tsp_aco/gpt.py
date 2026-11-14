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
    valid_directed = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(valid_directed):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid_directed, distances, np.inf)

    # Use the best available directed observation while producing
    # a symmetric prior for undirected stochastic sampling.
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

    row_order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)
    ranks = np.where(valid, ranks, np.inf)

    degree = valid.sum(axis=1).astype(float)
    density = float(valid.sum()) / max(float(n * (n - 1)), 1.0)
    sparsity = 1.0 - density

    # Robust node-specific scale.
    neighborhood_size = min(
        max(3, int(np.sqrt(max(n, 2))) + 1),
        n - 1,
    )
    nearest_values = np.take_along_axis(
        safe,
        row_order[:, :neighborhood_size],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_values)

    local_scale = (
        np.sum(
            np.where(nearest_valid, nearest_values, 0.0),
            axis=1,
        )
        / np.maximum(nearest_valid.sum(axis=1), 1.0)
    )
    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    scale_ratio = np.log(np.maximum(local_scale / global_scale, eps))
    heterogeneity = float(np.clip(np.std(scale_ratio) / 1.5, 0.0, 1.0))

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    local_distance = safe / np.maximum(pair_scale, eps)
    global_distance = safe / global_scale

    local_quality = np.exp(-np.clip(local_distance, 0.0, 50.0))
    global_quality = np.exp(-np.clip(global_distance, 0.0, 50.0))

    rank_temperature = max(float(min(4, n - 1)), 1.0)
    rank_quality = np.exp(
        -np.clip(ranks / rank_temperature, 0.0, 50.0)
    )
    reciprocal_rank = np.sqrt(rank_quality * rank_quality.T)

    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 50.0))
    )

    # Base quality: local scale remains dominant.
    local_weight = 0.56 - 0.08 * heterogeneity
    global_weight = 0.18 + 0.08 * heterogeneity
    affinity_weight = 1.0 - local_weight - global_weight

    quality = (
        local_weight * local_quality
        + global_weight * global_quality
        + affinity_weight * endpoint_affinity
    )
    quality = np.clip(quality, eps, None)

    # Candidate neighborhoods at two complementary resolutions.
    small_k = min(max(2, int(np.sqrt(max(n, 2)))), n - 1)
    large_k = min(max(small_k + 2, int(0.35 * (n - 1))), n - 1)

    small_mask = np.zeros((n, n), dtype=bool)
    large_mask = np.zeros((n, n), dtype=bool)

    small_mask[
        np.arange(n)[:, None],
        row_order[:, :small_k],
    ] = True
    large_mask[
        np.arange(n)[:, None],
        row_order[:, :large_k],
    ] = True

    small_mask &= valid
    large_mask &= valid

    small_degree = np.maximum(small_mask.sum(axis=1).astype(float), 1.0)
    large_degree = np.maximum(large_mask.sum(axis=1).astype(float), 1.0)

    small_overlap = (
        small_mask.astype(float) @ small_mask.astype(float).T
    ) / np.sqrt(small_degree[:, None] * small_degree[None, :])

    large_overlap = (
        large_mask.astype(float) @ large_mask.astype(float).T
    ) / np.sqrt(large_degree[:, None] * large_degree[None, :])

    small_overlap = np.clip(small_overlap, 0.0, 1.0)
    large_overlap = np.clip(large_overlap, 0.0, 1.0)

    # Neighborhood disagreement identifies candidate bridge edges.
    disagreement = (
        0.60 * (1.0 - small_overlap)
        + 0.40 * (1.0 - large_overlap)
    )

    mutual_support = np.sqrt(
        (
            small_mask.astype(float)
            + large_mask.astype(float)
        )
        / 2.0
        * (
            small_mask.astype(float)
            + large_mask.astype(float)
        ).T
        / 2.0
    )

    # Global diversity is a bounded novelty reward, not a hard balancing step.
    bridge_reward = disagreement * (
        0.45 + 0.55 * np.clip(mutual_support, 0.0, 1.0)
    )

    redundancy = (
        0.65 * small_overlap
        + 0.35 * large_overlap
    ) * np.clip(mutual_support, 0.0, 1.0)

    diversity_log = (
        (0.06 + 0.08 * sparsity) * bridge_reward
        - (0.035 + 0.035 * (1.0 - sparsity)) * redundancy
    )

    # Long-term potential.
    ordered_values = np.sort(
        np.where(valid, safe, np.inf),
        axis=1,
    )
    nearest = ordered_values[:, 0]
    second = ordered_values[:, min(1, n - 2)]

    regret = np.maximum(second - nearest, 0.0)
    regret = regret / np.maximum(local_scale, eps)
    regret = np.clip(regret, 0.0, 6.0)

    degree_reference = max(
        float(np.median(degree[degree > 0])),
        1.0,
    )
    scarcity = np.clip(
        degree_reference / np.maximum(degree, 1.0),
        0.5,
        2.5,
    )

    # A node is structurally fragile when its good-neighbor set is small
    # and the quality gap to its alternatives is large.
    candidate_width = np.maximum(large_degree, 1.0)
    flexibility = np.clip(
        candidate_width / max(float(large_k), 1.0),
        0.0,
        1.0,
    )
    fragility = np.clip(
        0.58 * (regret / 3.0)
        + 0.42 * (scarcity - 0.5) / 2.0,
        0.0,
        1.0,
    )

    endpoint_fragility = np.sqrt(
        (1.0 + fragility[:, None])
        * (1.0 + fragility[None, :])
    ) - 1.0

    # Penalize edges that consume the only strong option of both endpoints.
    shared_lock = np.sqrt(
        np.clip(
            (1.0 - flexibility[:, None])
            * (1.0 - flexibility[None, :]),
            0.0,
            1.0,
        )
    )

    # Prefer edges that provide a useful connection for fragile endpoints,
    # but keep the effect substantially weaker than distance quality.
    long_term_signal = np.clip(
        0.68 * endpoint_fragility
        + 0.32 * shared_lock
        - 0.20 * (1.0 - np.minimum(fragility[:, None], fragility[None, :])),
        -0.5,
        1.5,
    )

    long_term_weight = 0.08 + 0.08 * (
        0.55 * sparsity + 0.45 * heterogeneity
    )
    long_term_log = np.clip(
        long_term_weight * long_term_signal,
        -0.16,
        0.24,
    )

    rank_log = np.log(
        np.clip(
            0.64 + 0.36 * reciprocal_rank,
            eps,
            None,
        )
    )

    # Mild endpoint correction preserves diversity without hub flattening.
    row_quality = np.sum(
        np.where(valid, quality, 0.0),
        axis=1,
    )
    positive_row_quality = row_quality[row_quality > 0.0]

    if positive_row_quality.size:
        reference_quality = float(np.median(positive_row_quality))
        node_correction = np.sqrt(
            reference_quality / np.maximum(row_quality, eps)
        )
        node_correction = np.clip(node_correction, 0.88, 1.12)
        node_log = 0.5 * (
            np.log(node_correction[:, None])
            + np.log(node_correction[None, :])
        )
    else:
        node_log = np.zeros((n, n), dtype=float)

    log_scores = (
        np.log(quality)
        + rank_log
        + diversity_log
        + long_term_log
        + node_log
    )

    scores = np.exp(np.clip(log_scores, -50.0, 20.0))
    scores[~valid] = 0.0

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
