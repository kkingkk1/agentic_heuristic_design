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
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(
        valid,
        0.5 * (distances + distances.T),
        np.inf,
    )
    np.fill_diagonal(safe, np.inf)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)
    global_scale = max(global_scale, np.finfo(float).eps)

    order = np.argsort(safe, axis=1, kind="stable")
    rank = np.full((n, n), float(n), dtype=float)
    rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    valid_counts = np.sum(valid, axis=1).astype(float)

    def candidate_graph(k: int):
        k = min(max(int(k), 1), n - 1)
        mask = valid & (rank < float(k))
        chosen = np.where(mask, safe, np.inf)
        chosen_order = np.argsort(chosen, axis=1, kind="stable")[:, :k]
        chosen_distances = np.take_along_axis(
            safe,
            chosen_order,
            axis=1,
        )
        chosen_valid = np.isfinite(chosen_distances)
        counts = np.sum(chosen_valid, axis=1).astype(float)

        local_scale = np.sum(
            np.where(chosen_valid, chosen_distances, 0.0),
            axis=1,
        ) / np.maximum(counts, 1.0)
        local_scale = np.where(
            counts > 0.0,
            local_scale,
            global_scale,
        )
        local_scale = np.maximum(local_scale, np.finfo(float).eps)

        return mask, chosen_distances, chosen_valid, local_scale, counts

    short_k = min(2, n - 1)
    medium_k = min(4, n - 1)
    long_k = min(8, n - 1)

    short_graph, short_distances, short_valid, short_scale, short_counts = (
        candidate_graph(short_k)
    )
    medium_graph, medium_distances, medium_valid, medium_scale, medium_counts = (
        candidate_graph(medium_k)
    )
    long_graph, long_distances, long_valid, long_scale, long_counts = (
        candidate_graph(long_k)
    )

    local_scale = np.cbrt(
        np.maximum(short_scale, np.finfo(float).eps)
        * np.maximum(medium_scale, np.finfo(float).eps)
        * np.maximum(long_scale, np.finfo(float).eps)
    )

    normalized_distance = safe / np.maximum(
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        np.finfo(float).eps,
    )
    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    rank_score = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    short_mutual = short_graph & short_graph.T
    medium_mutual = medium_graph & medium_graph.T
    long_mutual = long_graph & long_graph.T

    # Soft candidate quality distribution and its entropy estimate
    # how many genuinely useful alternatives each endpoint retains.
    long_quality = np.where(
        long_valid,
        np.exp(
            -np.clip(
                long_distances / np.maximum(long_scale[:, None], np.finfo(float).eps),
                0.0,
                50.0,
            )
        ),
        0.0,
    )
    quality_sum = np.sum(long_quality, axis=1)
    probabilities = long_quality / np.maximum(
        quality_sum[:, None],
        np.finfo(float).eps,
    )
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0),
        axis=1,
    )
    entropy = entropy / np.log(max(float(long_k), 2.0))
    entropy = np.clip(entropy, 0.0, 1.0)

    sorted_quality = np.sort(long_quality, axis=1)
    strongest = sorted_quality[:, -1]
    second_strongest = (
        sorted_quality[:, -2]
        if long_k >= 2
        else np.zeros(n, dtype=float)
    )
    regret_reserve = second_strongest / np.maximum(
        strongest,
        np.finfo(float).eps,
    )
    availability = np.clip(
        (long_counts - 1.0) / max(float(long_k - 1), 1.0),
        0.0,
        1.0,
    )
    node_reserve = np.clip(
        0.45 * regret_reserve
        + 0.35 * entropy
        + 0.20 * availability,
        0.0,
        1.0,
    )
    endpoint_reserve = np.sqrt(
        node_reserve[:, None] * node_reserve[None, :]
    )

    # Edge competitiveness gates all global bonuses.
    edge_competitiveness = np.sqrt(
        np.exp(
            -np.clip(
                safe / np.maximum(local_scale[:, None], np.finfo(float).eps),
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                safe / np.maximum(local_scale[None, :], np.finfo(float).eps),
                0.0,
                50.0,
            )
        )
    )

    # Candidate neighborhoods at different horizons.
    short_float = short_graph.astype(float)
    medium_float = medium_graph.astype(float)
    long_float = long_graph.astype(float)
    combined = np.clip(short_float + medium_float + long_float, 0.0, 1.0)

    # Low neighborhood overlap indicates a possible cross-region connector.
    common = combined @ combined.T
    degree = np.sum(combined, axis=1)
    union_size = degree[:, None] + degree[None, :] - common
    bridge_signal = 1.0 - common / np.maximum(union_size, 1.0)
    bridge_signal = np.clip(
        bridge_signal * edge_competitiveness,
        0.0,
        1.0,
    )

    # Multi-hop continuation potential.
    second_hop = combined @ combined
    third_hop = second_hop @ combined
    second_mass = np.sum(np.minimum(second_hop, 1.0), axis=1)
    third_mass = np.sum(np.minimum(third_hop, 1.0), axis=1)

    second_mass /= max(float(n - 1), 1.0)
    third_mass /= max(float(n - 1), 1.0)

    continuation = np.sqrt(
        np.clip(
            (0.60 * second_mass + 0.40 * third_mass)[:, None]
            * (0.60 * second_mass + 0.40 * third_mass)[None, :],
            0.0,
            1.0,
        )
    )

    # A path-compatible edge gets additional value when its endpoints
    # expose complementary future neighborhoods.
    cross_continuation = (
        combined @ second_hop.T
        + combined.T @ second_hop
    )
    cross_continuation /= max(float(2 * n), 1.0)
    cross_continuation = np.clip(cross_continuation, 0.0, 1.0)

    persistent = (
        short_mutual
        & medium_mutual
        & long_mutual
    ).astype(float)

    medium_or_long = (medium_mutual | long_mutual).astype(float)
    long_only = (
        long_mutual
        & ~medium_mutual
        & ~short_mutual
    ).astype(float)

    opportunity_ratio = safe / np.maximum(
        np.minimum(local_scale[:, None], local_scale[None, :]),
        np.finfo(float).eps,
    )
    opportunity_penalty = 1.0 / (
        1.0
        + 0.065 * np.clip(opportunity_ratio - 1.0, 0.0, 10.0)
    )

    base_scores = (
        distance_score
        * (0.62 + 0.38 * rank_score)
        * (1.0 + 0.08 * short_mutual.astype(float))
        * (1.0 + 0.06 * medium_or_long)
        * (1.0 + 0.05 * persistent)
        * opportunity_penalty
    )
    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, np.finfo(float).eps)

    # Keep global signals deliberately small and apply them in log space.
    log_scores = np.log(base_scores)
    log_scores += 0.060 * endpoint_reserve * edge_competitiveness
    log_scores += 0.045 * bridge_signal * medium_or_long
    log_scores += 0.040 * continuation * edge_competitiveness
    log_scores += 0.030 * cross_continuation * edge_competitiveness
    log_scores += 0.025 * persistent
    log_scores -= 0.025 * long_only * (1.0 - edge_competitiveness)

    scores = np.exp(np.clip(log_scores, -60.0, 30.0))
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
