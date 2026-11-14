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
    directed_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    safe = np.minimum(
        np.where(directed_valid, distances, np.inf),
        np.where(directed_valid.T, distances.T, np.inf),
    )
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

    neighbor_count = min(4, n - 1)
    nearest_distances = np.partition(
        safe,
        neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

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
    endpoint_distance = safe * (
        1.0 / local_scale[:, None] + 1.0 / local_scale[None, :]
    )

    distance_score = np.exp(
        -np.minimum(normalized_distance, 50.0)
    )
    endpoint_affinity = np.exp(
        -0.5 * np.minimum(endpoint_distance, 100.0)
    )

    rank_window = min(8, n - 1)
    candidate_indices = np.argpartition(
        safe,
        rank_window - 1,
        axis=1,
    )[:, :rank_window]

    candidate_distances = np.take_along_axis(
        safe,
        candidate_indices,
        axis=1,
    )
    order = np.argsort(
        candidate_distances,
        axis=1,
        kind="stable",
    )
    neighbors = np.take_along_axis(
        candidate_indices,
        order,
        axis=1,
    )

    reciprocal_rank = np.full(
        (n, n),
        1.0 / (n + 1.0),
        dtype=float,
    )
    rank_values = 1.0 / (
        1.0 + np.arange(rank_window, dtype=float)
    )
    np.put_along_axis(
        reciprocal_rank,
        neighbors,
        np.broadcast_to(rank_values, (n, rank_window)),
        axis=1,
    )

    row_indices = np.arange(n)
    nearest_neighbor = neighbors[:, 0]

    mutual_nearest = (
        valid
        & (nearest_neighbor[:, None] == row_indices[None, :])
        & (nearest_neighbor[None, :] == row_indices[:, None])
    )

    top_k = min(2, n - 1)
    mutual_top = (
        valid
        & (reciprocal_rank > 1.0 / top_k)
        & (reciprocal_rank.T > 1.0 / top_k)
    )

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
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
