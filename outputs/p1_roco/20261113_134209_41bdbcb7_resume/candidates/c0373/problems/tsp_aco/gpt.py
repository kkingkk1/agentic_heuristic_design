import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    safe = np.minimum(
        np.where(directed_valid, distances, np.inf),
        np.where(directed_valid.T, distances.T, np.inf),
    )
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    indices = np.arange(n)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    nearest_neighbor = np.full(n, -1, dtype=int)

    local_k = min(4, n - 1)
    local_scale = np.full(n, np.nan, dtype=float)

    for i in range(n):
        valid_neighbors = row_order[i, valid[i, row_order[i]]]
        if valid_neighbors.size == 0:
            continue

        row_rank[i, valid_neighbors] = np.arange(
            valid_neighbors.size,
            dtype=float,
        )
        nearest_neighbor[i] = int(valid_neighbors[0])

        nearest_values = safe[i, valid_neighbors[:local_k]]
        local_scale[i] = float(np.median(nearest_values))

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 20.0)
    )

    bilateral_rank = 1.0 / np.sqrt(
        (1.0 + row_rank)
        * (1.0 + row_rank.T)
    )
    rank_factor = 0.82 + 0.18 * np.clip(
        bilateral_rank,
        0.0,
        1.0,
    )

    upper_valid = np.triu(valid, k=1)
    global_values = safe[upper_valid]

    if global_values.size <= 1:
        global_rank_score = np.ones_like(safe, dtype=float)
    else:
        sorted_global = np.sort(global_values)
        left_positions = np.searchsorted(
            sorted_global,
            safe,
            side="left",
        )
        global_rank_score = 1.0 - (
            left_positions / float(sorted_global.size - 1)
        )
        global_rank_score = np.clip(global_rank_score, 0.0, 1.0)

    global_rank_factor = 0.94 + 0.06 * global_rank_score

    mutual_nearest = (
        valid
        & (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    )

    top_k = min(3, n - 1)
    top_neighbors = valid & (row_rank < float(top_k))
    mutual_top_neighbors = top_neighbors & top_neighbors.T

    mutual_factor = (
        1.0
        + 0.10 * mutual_nearest.astype(float)
        + 0.04 * mutual_top_neighbors.astype(float)
    )

    candidate_popularity = top_neighbors.astype(float).sum(axis=0)
    popularity_reference = max(
        float(np.mean(candidate_popularity)),
        1.0,
    )
    popularity_ratio = np.clip(
        candidate_popularity / popularity_reference,
        0.0,
        2.0,
    )

    endpoint_popularity = 0.5 * (
        popularity_ratio[:, None]
        + popularity_ratio[None, :]
    )
    scarcity_factor = 1.0 + 0.06 * (
        1.0 - endpoint_popularity
    )

    scores = (
        distance_score
        * rank_factor
        * global_rank_factor
        * mutual_factor
        * scarcity_factor
    )

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
