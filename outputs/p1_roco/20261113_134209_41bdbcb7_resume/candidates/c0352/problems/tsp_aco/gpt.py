import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-priority matrix for stochastic TSP sampling."""
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

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    rank = np.full((n, n), float(n), dtype=float)
    rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    valid_counts = np.sum(valid, axis=1)

    def candidate_graph(k: int):
        k = min(k, n - 1)
        graph = valid & (rank < float(k))

        selected = np.where(graph, safe, np.inf)
        selected_order = np.argsort(selected, axis=1, kind="stable")[:, :k]
        selected_distances = np.take_along_axis(
            safe,
            selected_order,
            axis=1,
        )
        selected_valid = np.isfinite(selected_distances)

        local_scale = np.sum(
            np.where(selected_valid, selected_distances, 0.0),
            axis=1,
        ) / np.maximum(np.sum(selected_valid, axis=1), 1)

        local_scale = np.where(
            valid_counts > 0,
            local_scale,
            global_scale,
        )
        local_scale = np.maximum(local_scale, eps)

        return graph, selected_distances, selected_valid, local_scale

    short_k = min(3, n - 1)
    long_k = min(5, n - 1)

    short_graph, short_distances, short_valid, short_scale = (
        candidate_graph(short_k)
    )
    long_graph, long_distances, long_valid, long_scale = (
        candidate_graph(long_k)
    )

    local_scale = np.sqrt(
        np.maximum(short_scale, eps)
        * np.maximum(long_scale, eps)
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    rank_score = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    short_mutual = short_graph & short_graph.T
    long_mutual = long_graph & long_graph.T

    long_strength = np.where(
        long_valid,
        np.exp(
            -np.clip(
                long_distances / np.maximum(long_scale[:, None], eps),
                0.0,
                50.0,
            )
        ),
        0.0,
    )
    long_strength = np.sort(long_strength, axis=1)
    strongest = long_strength[:, -1]

    if long_k >= 2:
        second = long_strength[:, -2]
    else:
        second = np.zeros(n, dtype=float)

    reserve = np.clip(
        second / np.maximum(strongest, eps),
        0.0,
        1.0,
    )
    endpoint_reserve = np.sqrt(
        reserve[:, None] * reserve[None, :]
    )

    endpoint_competitiveness = np.sqrt(
        np.exp(
            -np.clip(
                safe / np.maximum(local_scale[:, None], eps),
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                safe / np.maximum(local_scale[None, :], eps),
                0.0,
                50.0,
            )
        )
    )

    combined_graph = (
        short_graph.astype(float)
        + long_graph.astype(float)
    )
    two_hop = combined_graph @ combined_graph.T
    coverage = np.clip(
        two_hop / max(float(2 * long_k), 1.0),
        0.0,
        1.0,
    )

    persistent_candidate = (
        short_graph
        & long_graph
        & short_graph.T
        & long_graph.T
    ).astype(float)

    neighborhood_disagreement = np.clip(
        1.0
        - 0.5
        * (
            np.clip(
                (short_graph.astype(float) @ short_graph.astype(float).T)
                / max(float(short_k), 1.0),
                0.0,
                1.0,
            )
            + np.clip(
                (long_graph.astype(float) @ long_graph.astype(float).T)
                / max(float(long_k), 1.0),
                0.0,
                1.0,
            )
        ),
        0.0,
        1.0,
    )

    opportunity_ratio = safe / np.maximum(
        np.minimum(local_scale[:, None], local_scale[None, :]),
        eps,
    )
    opportunity_penalty = 1.0 / (
        1.0
        + 0.055 * np.clip(opportunity_ratio - 1.0, 0.0, 6.0)
    )

    base_scores = (
        distance_score
        * endpoint_competitiveness
        * (0.62 + 0.38 * rank_score)
        * (1.0 + 0.27 * short_mutual.astype(float))
        * (1.0 + 0.08 * long_mutual.astype(float))
        * opportunity_penalty
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    log_scores = np.log(base_scores)

    log_scores += (
        0.030
        * endpoint_reserve
        * long_graph
        * long_graph.T
        * endpoint_competitiveness
    )
    log_scores += (
        0.016
        * neighborhood_disagreement
        * endpoint_competitiveness
        * short_mutual.astype(float)
    )
    log_scores += (
        0.014
        * coverage
        * endpoint_competitiveness
        * (
            0.65 * endpoint_reserve
            + 0.35 * persistent_candidate
        )
    )
    log_scores += (
        0.013
        * endpoint_reserve
        * persistent_candidate
    )

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
