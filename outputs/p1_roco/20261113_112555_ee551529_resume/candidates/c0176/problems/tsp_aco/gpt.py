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
    valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    neighborhood_size = min(4, n - 1)
    nearest_distances = np.partition(
        safe,
        kth=neighborhood_size - 1,
        axis=1,
    )[:, :neighborhood_size]

    local_scale = np.mean(nearest_distances, axis=1)
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
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    row_endpoint_score = np.exp(
        -np.clip(safe / local_scale[:, None], 0.0, 50.0)
    )
    col_endpoint_score = np.exp(
        -np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )
    endpoint_affinity = np.sqrt(
        row_endpoint_score * col_endpoint_score
    )

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

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

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
    reciprocal_rank = np.where(
        reciprocal_rank > 0.0,
        reciprocal_rank,
        0.0,
    )

    nearest_row = row_neighbors[:, 0]
    nearest_col = col_neighbors[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

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

    isolation_ratio = pair_scale / (
        safe + global_scale
    )
    isolation_bonus = 1.0 + 0.20 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_affinity
        * (0.50 + 0.50 * reciprocal_rank)
        * (
            1.0
            + 0.30 * mutual_nearest
            + 0.10 * mutual_top
        )
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
