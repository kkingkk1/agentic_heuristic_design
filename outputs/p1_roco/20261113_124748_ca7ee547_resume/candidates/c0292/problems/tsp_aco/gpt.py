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
    raw_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(raw_valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(raw_valid, distances, np.inf)
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
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    valid_counts = np.sum(valid, axis=1)
    neighborhood_k = min(5, n - 1)

    neighborhood = np.take_along_axis(
        safe,
        row_order[:, :neighborhood_k],
        axis=1,
    )
    neighborhood_valid = np.isfinite(neighborhood)

    local_scale = np.sum(
        np.where(neighborhood_valid, neighborhood, 0.0),
        axis=1,
    ) / np.maximum(np.sum(neighborhood_valid, axis=1), 1)

    local_scale = np.where(valid_counts > 0, local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    distance_log = -np.clip(normalized_distance, 0.0, 50.0)

    row_affinity_log = -np.clip(
        safe / local_scale[:, None],
        0.0,
        50.0,
    )
    col_affinity_log = -np.clip(
        safe / local_scale[None, :],
        0.0,
        50.0,
    )
    endpoint_affinity_log = 0.5 * (row_affinity_log + col_affinity_log)

    rank_log = -0.5 * (
        np.log1p(row_rank) + np.log1p(col_rank)
    )

    top_k = min(3, n - 1)
    row_top = valid & (row_rank < float(top_k))
    col_top = valid & (col_rank < float(top_k))

    mutual_top = row_top & col_top

    nearest_row = row_order[:, 0]
    nearest_col = col_order[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    )

    competitive_k = min(5, n - 1)
    competitive = (
        valid
        & (row_rank < float(competitive_k))
        & (col_rank < float(competitive_k))
    )

    top_adjacency = row_top.astype(float)
    shared_neighbors = top_adjacency @ top_adjacency.T
    two_hop_support = np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
    )

    local_strength = np.where(
        valid,
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        / np.sqrt(1.0 + row_rank),
        0.0,
    )

    strongest = np.max(local_strength, axis=1)
    total_strength = np.sum(local_strength, axis=1)

    concentration = strongest / np.maximum(total_strength, eps)
    concentration = np.clip(concentration, 0.0, 1.0)

    sorted_strength = np.sort(local_strength, axis=1)
    second_strength = sorted_strength[:, -2] if n >= 2 else np.zeros(n)

    alternative_ratio = second_strength / np.maximum(strongest, eps)
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)

    local_density = np.clip(
        local_scale / np.maximum(global_scale, eps),
        0.0,
        3.0,
    )
    sparse_nodes = local_density / (1.0 + local_density)

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + col_rank), 0.0, 1.0)
    )

    opportunity_ratio = safe / np.maximum(
        np.minimum(local_scale[:, None], local_scale[None, :]),
        eps,
    )
    opportunity_penalty = -np.log1p(
        0.14 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    mutual_log_bonus = (
        0.24 * mutual_nearest.astype(float)
        + 0.09 * mutual_top.astype(float)
    )

    two_hop_log_bonus = (
        0.07
        * two_hop_support
        * competitive.astype(float)
    )

    concentration_penalty = (
        0.075
        * np.sqrt(concentration[:, None] * concentration[None, :])
        * edge_topness
    )

    complementary_bonus = (
        0.065
        * np.sqrt(
            alternative_ratio[:, None]
            * alternative_ratio[None, :]
        )
        * competitive.astype(float)
    )

    sparse_alternative_bonus = (
        0.045
        * np.sqrt(
            sparse_nodes[:, None] * sparse_nodes[None, :]
        )
        * (1.0 - edge_topness)
    )

    isolation_bonus = 0.12 * np.clip(
        pair_scale / np.maximum(safe + global_scale, eps),
        0.0,
        1.0,
    )

    log_scores = (
        distance_log
        + endpoint_affinity_log
        + 0.34 * rank_log
        + opportunity_penalty
        + mutual_log_bonus
        + two_hop_log_bonus
        + isolation_bonus
        + complementary_bonus
        + sparse_alternative_bonus
        - concentration_penalty
    )

    log_scores[~valid] = -np.inf
    finite_log_scores = np.where(valid, log_scores, -np.inf)

    max_log = np.max(finite_log_scores)
    if not np.isfinite(max_log):
        return np.zeros((n, n), dtype=float)

    scores = np.exp(np.clip(log_scores - max_log, -60.0, 0.0))
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
