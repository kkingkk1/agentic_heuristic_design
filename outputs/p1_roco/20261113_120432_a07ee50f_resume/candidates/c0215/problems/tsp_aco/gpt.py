import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    raw_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(raw_valid):
        return scores

    safe = np.where(raw_valid, distances, np.inf)

    # Construct a symmetric finite-distance representation.
    safe = np.minimum(safe, safe.T)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe) & ~diagonal
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Estimate endpoint-local scales from the nearest valid neighbors.
    neighborhood_size = min(4, n - 1)
    nearest_values = np.partition(
        safe,
        kth=neighborhood_size - 1,
        axis=1,
    )[:, :neighborhood_size]

    neighborhood_valid = np.isfinite(nearest_values)
    neighborhood_count = neighborhood_valid.sum(axis=1)

    local_scale = np.sum(
        np.where(neighborhood_valid, nearest_values, 0.0),
        axis=1,
    ) / np.maximum(neighborhood_count, 1)

    local_scale = np.where(
        neighborhood_count > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized_distance = safe / np.maximum(pair_scale, eps)
        distance_score = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0)
        )

        endpoint_row_score = np.exp(
            -np.clip(safe / local_scale[:, None], 0.0, 50.0)
        )
        endpoint_col_score = np.exp(
            -np.clip(safe / local_scale[None, :], 0.0, 50.0)
        )

    endpoint_affinity = np.sqrt(
        endpoint_row_score * endpoint_col_score
    )

    # Obtain bounded reciprocal ranks for both endpoints.
    rank_window = min(8, n - 1)

    row_candidates = np.argpartition(
        safe,
        kth=rank_window - 1,
        axis=1,
    )[:, :rank_window]
    row_candidate_distances = np.take_along_axis(
        safe,
        row_candidates,
        axis=1,
    )
    row_order = np.argsort(
        row_candidate_distances,
        axis=1,
        kind="stable",
    )
    row_neighbors = np.take_along_axis(
        row_candidates,
        row_order,
        axis=1,
    )

    col_candidates = np.argpartition(
        safe,
        kth=rank_window - 1,
        axis=0,
    )[:rank_window, :]
    col_candidate_distances = np.take_along_axis(
        safe,
        col_candidates,
        axis=0,
    )
    col_order = np.argsort(
        col_candidate_distances,
        axis=0,
        kind="stable",
    )
    col_neighbors = np.take_along_axis(
        col_candidates,
        col_order,
        axis=0,
    )

    row_rank = np.full((n, n), float(rank_window), dtype=float)
    col_rank = np.full((n, n), float(rank_window), dtype=float)

    row_rank_values = np.broadcast_to(
        np.arange(rank_window, dtype=float),
        (n, rank_window),
    )
    col_rank_values = np.broadcast_to(
        np.arange(rank_window, dtype=float)[:, None],
        (rank_window, n),
    )

    np.put_along_axis(
        row_rank,
        row_neighbors,
        row_rank_values,
        axis=1,
    )
    np.put_along_axis(
        col_rank,
        col_neighbors,
        col_rank_values,
        axis=0,
    )

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    reciprocal_rank = np.where(valid, reciprocal_rank, 0.0)

    # Mutual nearest-neighbor structure.
    nearest_row = row_neighbors[:, 0]
    nearest_col = col_neighbors[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Mutual top-two local candidate structure.
    top_k = min(2, n - 1)
    row_top = np.zeros((n, n), dtype=bool)
    col_top = np.zeros((n, n), dtype=bool)

    np.put_along_axis(
        row_top,
        row_neighbors[:, :top_k],
        True,
        axis=1,
    )
    np.put_along_axis(
        col_top,
        col_neighbors[:top_k, :],
        True,
        axis=0,
    )

    mutual_top = (row_top & col_top & valid).astype(float)

    # Bounded local competition penalty.
    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe / np.maximum(endpoint_best, eps)
    opportunity_penalty = 1.0 / (
        1.0
        + 0.16 * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    # Mild isolation bonus for edges that are strong relative to endpoint scales.
    isolation_ratio = pair_scale / (
        safe + global_scale
    )
    isolation_bonus = 1.0 + 0.16 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_affinity
        * (0.56 + 0.44 * reciprocal_rank)
        * (1.0 + 0.28 * mutual_nearest)
        * (1.0 + 0.10 * mutual_top)
        * opportunity_penalty
        * isolation_bonus
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
