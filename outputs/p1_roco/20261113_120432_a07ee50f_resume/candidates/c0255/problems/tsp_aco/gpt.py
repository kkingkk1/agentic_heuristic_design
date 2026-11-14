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
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    safe = np.where(directed_valid, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    values = safe[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(values)), 1.0)
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(safe, order, axis=1)
    finite_sorted = np.isfinite(sorted_distances)

    degree = valid.sum(axis=1).astype(float)
    density = float(valid.sum()) / max(float(n * (n - 1)), 1.0)
    sparsity = 1.0 - density

    # Use a trimmed local scale based on at most the two closest usable neighbors.
    k_scale = min(3, n - 1)
    nearest = sorted_distances[:, :k_scale]
    if k_scale >= 3:
        scale_values = nearest[:, :2]
    else:
        scale_values = nearest

    scale_valid = np.isfinite(scale_values)
    scale_sum = np.sum(np.where(scale_valid, scale_values, 0.0), axis=1)
    scale_count = scale_valid.sum(axis=1)

    local_scale = np.divide(
        scale_sum,
        np.maximum(scale_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=scale_count > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    positive_local = local_scale[local_scale > eps]
    fallback_scale = (
        float(np.median(positive_local))
        if positive_local.size
        else global_scale
    )
    fallback_scale = max(fallback_scale, eps)

    denominator = np.maximum(local_scale, fallback_scale * 1e-6)
    pair_scale = np.sqrt(
        np.maximum(denominator[:, None] * denominator[None, :], eps)
    )

    normalized_distance = np.divide(
        safe,
        pair_scale,
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )
    clipped_distance = np.clip(normalized_distance, 0.0, 1e6)

    # Distance quality remains the dominant signal.
    distance_signal = -np.log1p(clipped_distance)

    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    rank_signal = -0.5 * (
        np.log1p(row_rank) + np.log1p(row_rank.T)
    )
    rank_signal = np.maximum(rank_signal, -np.log1p(float(n)))

    endpoint_ratio_i = np.divide(
        safe,
        denominator[:, None],
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )
    endpoint_ratio_j = np.divide(
        safe,
        denominator[None, :],
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )

    endpoint_affinity = -0.5 * (
        np.log1p(np.clip(endpoint_ratio_i, 0.0, 1e6))
        + np.log1p(np.clip(endpoint_ratio_j, 0.0, 1e6))
    )

    # Candidate-neighborhood disagreement supplies a bounded exploration signal.
    small_k = min(max(2, int(np.sqrt(max(n, 2)))), n - 1)
    large_k = min(
        max(small_k + 2, int(0.35 * (n - 1))),
        n - 1,
    )

    small_neighbors = np.zeros((n, n), dtype=bool)
    large_neighbors = np.zeros((n, n), dtype=bool)

    small_neighbors[
        np.arange(n)[:, None],
        order[:, :small_k],
    ] = True
    large_neighbors[
        np.arange(n)[:, None],
        order[:, :large_k],
    ] = True

    small_neighbors &= valid
    large_neighbors &= valid

    small_degree = np.maximum(
        small_neighbors.sum(axis=1).astype(float),
        1.0,
    )
    large_degree = np.maximum(
        large_neighbors.sum(axis=1).astype(float),
        1.0,
    )

    small_overlap = (
        small_neighbors.astype(float) @ small_neighbors.astype(float).T
    ) / np.sqrt(small_degree[:, None] * small_degree[None, :])

    large_overlap = (
        large_neighbors.astype(float) @ large_neighbors.astype(float).T
    ) / np.sqrt(large_degree[:, None] * large_degree[None, :])

    small_overlap = np.clip(small_overlap, 0.0, 1.0)
    large_overlap = np.clip(large_overlap, 0.0, 1.0)

    disagreement = (
        0.60 * (1.0 - small_overlap)
        + 0.40 * (1.0 - large_overlap)
    )

    mutual_support = np.sqrt(
        np.clip(
            0.5 * (small_neighbors.astype(float) + large_neighbors.astype(float))
            * 0.5 * (
                small_neighbors.astype(float).T
                + large_neighbors.astype(float).T
            ),
            0.0,
            1.0,
        )
    )

    bridge_reward = disagreement * (
        0.45 + 0.55 * mutual_support
    )
    redundancy = (
        0.65 * small_overlap
        + 0.35 * large_overlap
    ) * mutual_support

    diversity_log = (
        (0.045 + 0.065 * sparsity) * bridge_reward
        - (0.025 + 0.025 * (1.0 - sparsity)) * redundancy
    )

    # Long-term potential from endpoint regret, scarcity, and flexibility.
    ordered_values = np.sort(
        np.where(valid, safe, np.inf),
        axis=1,
    )
    nearest = ordered_values[:, 0]
    second_nearest = ordered_values[:, min(1, n - 2)]

    second_nearest = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        np.maximum(local_scale, global_scale),
    )

    regret = np.maximum(second_nearest - nearest, 0.0)
    regret = np.clip(
        regret / np.maximum(local_scale, eps),
        0.0,
        6.0,
    )

    degree_reference = max(
        float(np.median(degree[degree > 0])),
        1.0,
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
        0.58 * (regret / 3.0)
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
        0.68 * endpoint_fragility
        + 0.32 * shared_lock
        - 0.20 * (
            1.0 - np.minimum(
                fragility[:, None],
                fragility[None, :],
            )
        ),
        -0.5,
        1.5,
    )

    long_term_weight = 0.06 + 0.07 * (
        0.55 * sparsity + 0.45 * np.clip(
            np.std(np.log(np.maximum(local_scale / global_scale, eps))) / 1.5,
            0.0,
            1.0,
        )
    )
    long_term_log = np.clip(
        long_term_weight * long_term_signal,
        -0.14,
        0.20,
    )

    # Mild protection for genuinely sparse nodes.
    sparse_endpoint = (
        (degree[:, None] <= 2)
        | (degree[None, :] <= 2)
    )
    sparse_protection = (
        0.07
        * sparse_endpoint.astype(float)
        * np.exp(-np.clip(normalized_distance, 0.0, 10.0))
    )

    # Gentle opportunity-cost penalty, relative to the weaker endpoint alternative.
    opportunity_denominator = np.minimum(
        np.maximum(second_nearest[:, None], eps),
        np.maximum(second_nearest[None, :], eps),
    )
    opportunity_ratio = np.divide(
        safe,
        opportunity_denominator,
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )
    opportunity_penalty = -0.06 * np.clip(
        opportunity_ratio - 1.30,
        0.0,
        3.0,
    )

    log_scores = (
        0.52 * distance_signal
        + 0.22 * rank_signal
        + 0.18 * endpoint_affinity
        + diversity_log
        + long_term_log
        + opportunity_penalty
        + sparse_protection
    )
    log_scores[~valid] = -np.inf

    valid_scores = log_scores[valid]
    if valid_scores.size == 0:
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(valid_scores))
    scores = np.zeros((n, n), dtype=float)
    scores[valid] = np.exp(
        np.clip(log_scores[valid] - maximum, -50.0, 0.0)
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
