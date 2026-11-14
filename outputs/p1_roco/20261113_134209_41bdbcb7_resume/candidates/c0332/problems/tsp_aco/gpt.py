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

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return result

    directed = np.where(raw_valid, distances, np.inf)

    # Convert directed input into a symmetric finite undirected graph.
    safe = np.minimum(directed, directed.T)
    valid = np.isfinite(safe) & ~diagonal
    safe[~valid] = np.inf

    if not np.any(valid):
        return result

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    # Stable bidirectional ranks over valid edges.
    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_positions = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    np.put_along_axis(row_rank, order, row_positions, axis=1)
    row_rank[~valid] = float(n)

    col_rank = row_rank.T
    rank_agreement = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    # Local scale from the closest valid neighbors only.
    neighborhood_size = min(5, n - 1)
    nearest_indices = order[:, :neighborhood_size]
    nearest_distances = np.take_along_axis(
        safe,
        nearest_indices,
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_distances)

    local_scale = np.divide(
        np.where(nearest_valid, nearest_distances, 0.0).sum(axis=1),
        np.maximum(nearest_valid.sum(axis=1), 1),
    )
    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = np.divide(
        safe,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    endpoint_distance = np.divide(
        safe,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_distance_t = np.divide(
        safe,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    # Main distance evidence: global-relative and endpoint-relative closeness.
    log_scores = (
        -1.18 * normalized_distance
        -0.42 * np.clip(endpoint_distance, 0.0, 50.0)
        -0.42 * np.clip(endpoint_distance_t, 0.0, 50.0)
    )

    # Continuous bidirectional rank evidence.
    log_scores += 0.62 * np.log(
        0.55 + 0.45 * rank_agreement
    )

    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        nearest_indices[:, :top_k],
    ] = nearest_valid[:, :top_k]

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    )

    nearest = order[:, 0]
    has_nearest = np.isfinite(safe[np.arange(n), nearest])
    mutual_nearest = (
        valid
        & has_nearest[:, None]
        & has_nearest[None, :]
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )

    log_scores += np.log(
        1.0
        + 0.24 * mutual_top.astype(float)
        + 0.34 * mutual_nearest.astype(float)
    )

    # Mild opportunity-cost penalty for edges that are poor for an endpoint.
    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = np.divide(
        safe,
        np.maximum(endpoint_best, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    opportunity_penalty = 0.12 * np.clip(
        opportunity_ratio - 1.0,
        0.0,
        6.0,
    )
    log_scores -= opportunity_penalty

    # Reward edges incident to sparse nodes, without overriding distance evidence.
    degree = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)
    log_scores += 0.10 * scarcity_pair

    # Prefer locally competitive edges and mildly reduce overly popular endpoints.
    competitive_k = min(4, n - 1)
    locally_competitive = (
        valid
        & (row_rank < competitive_k)
        & (col_rank < competitive_k)
    )

    popularity = top_neighbors.sum(axis=0).astype(float)
    popularity_reference = max(float(np.mean(popularity)), 1.0)
    popularity_ratio = np.clip(
        popularity / popularity_reference,
        0.0,
        3.0,
    )
    popularity_penalty = 0.055 * np.sqrt(
        (1.0 + popularity_ratio[:, None])
        * (1.0 + popularity_ratio[None, :])
    )
    popularity_penalty = np.where(
        locally_competitive,
        popularity_penalty,
        0.5 * popularity_penalty,
    )
    log_scores -= popularity_penalty

    # Mild bonus for local-neighborhood agreement.
    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )
    overlap_bonus = 0.045 * np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
    )
    log_scores += overlap_bonus * locally_competitive.astype(float)

    log_scores[~valid] = -np.inf

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
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
