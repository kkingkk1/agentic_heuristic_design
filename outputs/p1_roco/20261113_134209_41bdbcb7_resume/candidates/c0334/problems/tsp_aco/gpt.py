import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return result

    eps = np.finfo(float).eps
    directed = np.where(directed_valid, distances, np.inf)

    # Convert asymmetric input into a symmetric graph.
    safe = np.minimum(directed, directed.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return result

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Stable bidirectional ranks.
    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    col_rank = row_rank.T

    # Local scale from the nearest valid neighbors only.
    neighborhood_size = min(5, n - 1)
    nearest = np.take_along_axis(
        safe,
        order[:, :neighborhood_size],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest)

    local_scale = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    ) / np.maximum(nearest_valid.sum(axis=1), 1)

    local_scale = np.where(
        nearest_valid.sum(axis=1) > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        safe,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=np.isfinite(safe),
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    distance_score = np.exp(
        -np.clip(relative_distance ** 1.18, 0.0, 50.0)
    )

    endpoint_score = np.sqrt(
        np.exp(
            -np.clip(
                (safe / local_scale[:, None]) ** 1.08,
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                (safe / local_scale[None, :]) ** 1.08,
                0.0,
                50.0,
            )
        )
    )

    rank_agreement = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    # Reciprocal nearest-neighbor and reciprocal top-neighbor structure.
    nearest_node = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        valid
        & (nearest_node[:, None] == indices[None, :])
        & (nearest_node[None, :] == indices[:, None])
    ).astype(float)

    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True
    top_neighbors &= valid

    mutual_top = (
        top_neighbors
        & top_neighbors.T
    ).astype(float)

    competitive_k = min(5, n - 1)
    locally_competitive = (
        valid
        & (row_rank < float(competitive_k))
        & (col_rank < float(competitive_k))
    ).astype(float)

    # Opportunity cost relative to each endpoint's best local alternative.
    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = np.divide(
        safe,
        np.maximum(endpoint_best, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=np.isfinite(safe),
    )

    opportunity_penalty = np.exp(
        -0.16 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    # Shared-neighborhood structure provides a restrained two-hop signal.
    top_float = top_neighbors.astype(float)
    shared_neighbors = top_float @ top_float.T
    union_neighbors = (
        top_float.sum(axis=1)[:, None]
        + top_float.sum(axis=1)[None, :]
        - shared_neighbors
    )

    overlap = shared_neighbors / np.maximum(union_neighbors, 1.0)

    two_hop_bonus = (
        1.0
        + 0.06
        * np.clip(
            shared_neighbors / max(float(top_k), 1.0),
            0.0,
            1.0,
        )
        * locally_competitive
    )

    diversity_bonus = (
        1.0
        + 0.08
        * np.clip(1.0 - overlap, 0.0, 1.0)
        * locally_competitive
    )

    # Penalize edges whose endpoints are overly popular candidate targets.
    popularity = top_float.sum(axis=0)
    popularity_reference = max(float(np.mean(popularity)), 1.0)
    popularity_ratio = np.clip(
        popularity / popularity_reference,
        0.0,
        3.0,
    )

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + 0.13 * popularity_ratio[:, None])
        * (1.0 + 0.13 * popularity_ratio[None, :])
    )

    # Preserve alternatives for sparse or locally isolated endpoints.
    valid_degree = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(valid_degree, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), 1.0)

    isolation_ratio = np.divide(
        pair_scale,
        safe + global_scale,
        out=np.zeros((n, n), dtype=float),
        where=np.isfinite(safe),
    )
    isolation_bonus = 1.0 + 0.12 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    mutual_bonus = (
        1.0
        + 0.24 * mutual_nearest
        + 0.08 * mutual_top
    )

    base_scores = (
        distance_score
        * endpoint_score
        * (0.56 + 0.44 * rank_agreement)
        * mutual_bonus
        * opportunity_penalty
        * two_hop_bonus
        * diversity_bonus
        * concentration_penalty
        * (0.90 + 0.20 * scarcity_pair)
        * isolation_bonus
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    # Small log-space structural correction keeps distance dominance.
    log_scores = np.log(base_scores)
    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + col_rank), 0.0, 1.0)
    )

    log_scores += (
        0.035
        * (1.0 - overlap)
        * locally_competitive
    )
    log_scores -= (
        0.045
        * edge_topness
        * np.sqrt(
            popularity_ratio[:, None] * popularity_ratio[None, :]
        )
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
