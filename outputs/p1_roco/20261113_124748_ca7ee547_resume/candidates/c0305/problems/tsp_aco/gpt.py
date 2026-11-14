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

    if not np.any(directed_valid):
        return np.zeros((n, n), dtype=float)

    # An edge is usable only when both directions are valid. This gives all
    # subsequent structural signals a common symmetric support.
    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, 0.5 * (distances + distances.T), np.inf)

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
    candidate_k = min(5, n - 1)

    candidate_order = row_order[:, :candidate_k]
    candidate_distances = np.take_along_axis(
        safe,
        candidate_order,
        axis=1,
    )
    candidate_valid = np.isfinite(candidate_distances)

    local_scale = np.sum(
        np.where(candidate_valid, candidate_distances, 0.0),
        axis=1,
    ) / np.maximum(np.sum(candidate_valid, axis=1), 1)

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

    # Stable rank preference, with a softer tail so that useful alternatives
    # remain available to stochastic sampling.
    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )

    # Local candidate graph.
    candidate_mask = valid & (row_rank < float(candidate_k))
    candidate_graph = candidate_mask.astype(float)

    mutual_candidate = (
        candidate_mask
        & candidate_mask.T
    ).astype(float)

    nearest = row_order[:, 0]
    nearest_valid = np.isfinite(safe[np.arange(n), nearest])
    mutual_nearest = (
        valid
        & nearest_valid[:, None]
        & (nearest[:, None] == np.arange(n)[None, :])
        & nearest_valid[None, :]
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Estimate the cost of committing an endpoint to its strongest edge.
    # The second candidate creates a regret/reserve signal without requiring
    # entropy, which can duplicate local-neighborhood information.
    candidate_strength = np.where(
        candidate_valid,
        np.exp(
            -np.clip(
                candidate_distances
                / np.maximum(local_scale[:, None], eps),
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

    reserve_ratio = second / np.maximum(strongest, eps)
    reserve_ratio = np.clip(reserve_ratio, 0.0, 1.0)

    # A node with low reserve is fragile: edges incident to it should be
    # favored only when they are genuinely competitive. A node with high
    # reserve can safely support a bridge or diversification edge.
    endpoint_reserve = np.sqrt(
        reserve_ratio[:, None] * reserve_ratio[None, :]
    )

    edge_competitiveness = np.sqrt(
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

    # Candidate-neighborhood overlap is a direct, interpretable bridge
    # measure. Low overlap indicates that an edge can connect distinct
    # local basins; the candidate and competitiveness gates keep it bounded.
    shared_candidates = candidate_graph @ candidate_graph.T
    neighborhood_overlap = shared_candidates / max(float(candidate_k), 1.0)
    neighborhood_overlap = np.clip(neighborhood_overlap, 0.0, 1.0)

    neighborhood_difference = 1.0 - neighborhood_overlap

    bridge_signal = (
        neighborhood_difference
        * edge_competitiveness
        * np.sqrt(
            np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
            * np.clip(1.0 / (1.0 + row_rank.T), 0.0, 1.0)
        )
        * candidate_mask
        * candidate_mask.T
    )

    # Two-hop coverage: an edge is useful when its endpoints collectively
    # expose many distinct candidate nodes. This promotes eventual tour
    # coverage while remaining local and bounded.
    combined_neighbors = candidate_graph + candidate_graph.T
    two_hop_coverage = combined_neighbors @ combined_neighbors.T
    coverage_scale = max(float(2 * candidate_k), 1.0)
    coverage_signal = np.clip(
        two_hop_coverage / coverage_scale,
        0.0,
        1.0,
    )

    # Mild opportunity penalty preserves alternatives but avoids assigning
    # excessive probability to edges much worse than both endpoint scales.
    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_best, eps)
    opportunity_penalty = 1.0 / (
        1.0
        + 0.06 * np.clip(opportunity_ratio - 1.0, 0.0, 6.0)
    )

    # Delayed-commitment bonus: alternatives are rewarded when both endpoints
    # retain reserve capacity. Strong nearest-neighbor edges receive less of
    # this bonus, preventing diversity terms from replacing distance quality.
    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + row_rank.T), 0.0, 1.0)
    )

    delayed_commitment = (
        endpoint_reserve
        * (0.35 + 0.65 * (1.0 - edge_topness))
        * edge_competitiveness
        * candidate_mask
        * candidate_mask.T
    )

    base_scores = (
        distance_score
        * (0.60 + 0.40 * rank_score)
        * (1.0 + 0.28 * mutual_nearest)
        * (1.0 + 0.14 * mutual_candidate)
        * opportunity_penalty
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    log_scores = np.log(base_scores)

    # All long-term signals are deliberately small and additive in log space.
    # This keeps the shortest reliable edges dominant while allowing global
    # coverage to break near-ties.
    log_scores += 0.085 * delayed_commitment
    log_scores += 0.050 * bridge_signal
    log_scores += 0.035 * coverage_signal * edge_competitiveness
    log_scores += 0.025 * endpoint_reserve * mutual_candidate

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
