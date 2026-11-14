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

    # Use only edges valid in both directions so every structural signal
    # shares the same symmetric support.
    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(
        valid,
        0.5 * (distances + distances.T),
        np.inf,
    )

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    valid_counts = np.sum(valid, axis=1)
    candidate_k = min(4, n - 1)

    candidate_order = row_order[:, :candidate_k]
    candidate_distances = np.take_along_axis(
        safe,
        candidate_order,
        axis=1,
    )
    candidate_valid = np.isfinite(candidate_distances)

    local_scale = (
        np.sum(
            np.where(candidate_valid, candidate_distances, 0.0),
            axis=1,
        )
        / np.maximum(np.sum(candidate_valid, axis=1), 1)
    )
    local_scale = np.where(
        valid_counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )

    candidate_mask = valid & (row_rank < float(candidate_k))
    mutual_candidate = (
        candidate_mask & candidate_mask.T
    ).astype(float)

    nearest = row_order[:, 0]
    nearest_valid = np.isfinite(safe[np.arange(n), nearest])
    mutual_nearest = (
        valid
        & nearest_valid[:, None]
        & nearest_valid[None, :]
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Reserve signal: retain alternatives when neither endpoint has a
    # completely dominant first choice.
    candidate_strength = np.where(
        candidate_valid,
        np.exp(
            -np.clip(
                candidate_distances / np.maximum(local_scale[:, None], eps),
                0.0,
                50.0,
            )
        ),
        0.0,
    )

    sorted_strength = np.sort(candidate_strength, axis=1)
    strongest = sorted_strength[:, -1]
    second = (
        sorted_strength[:, -2]
        if candidate_k >= 2
        else np.zeros(n, dtype=float)
    )

    reserve_ratio = np.clip(
        second / np.maximum(strongest, eps),
        0.0,
        1.0,
    )
    endpoint_reserve = np.sqrt(
        reserve_ratio[:, None] * reserve_ratio[None, :]
    )

    endpoint_affinity = np.sqrt(
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

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_best, eps)
    opportunity_penalty = 1.0 / (
        1.0
        + 0.07 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    # Mild bridge signal from candidate-neighborhood differences. It is
    # gated by mutual candidacy and endpoint affinity.
    candidate_graph = candidate_mask.astype(float)
    shared_candidates = candidate_graph @ candidate_graph.T
    neighborhood_overlap = np.clip(
        shared_candidates / max(float(candidate_k), 1.0),
        0.0,
        1.0,
    )
    neighborhood_difference = 1.0 - neighborhood_overlap

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + row_rank.T), 0.0, 1.0)
    )

    bridge_signal = (
        neighborhood_difference
        * endpoint_affinity
        * edge_topness
        * candidate_mask
        * candidate_mask.T
    )

    # Bounded coverage signal encourages edges whose endpoint neighborhoods
    # expose distinct alternatives.
    combined_neighbors = candidate_graph + candidate_graph.T
    two_hop_coverage = combined_neighbors @ combined_neighbors.T
    coverage_signal = np.clip(
        two_hop_coverage / max(float(2 * candidate_k), 1.0),
        0.0,
        1.0,
    )

    base_scores = (
        distance_score
        * endpoint_affinity
        * (0.55 + 0.45 * rank_score)
        * (
            1.0
            + 0.36 * mutual_nearest
            + 0.15 * mutual_candidate
        )
        * opportunity_penalty
        * (
            1.0
            + 0.08 * endpoint_reserve * mutual_candidate
        )
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    log_scores = np.log(base_scores)

    # Keep exploratory terms small so they resolve near ties instead of
    # replacing strong short edges.
    log_scores += (
        0.055
        * endpoint_reserve
        * (0.35 + 0.65 * (1.0 - edge_topness))
        * endpoint_affinity
        * candidate_mask
        * candidate_mask.T
    )
    log_scores += 0.030 * bridge_signal
    log_scores += 0.020 * coverage_signal * endpoint_affinity

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
