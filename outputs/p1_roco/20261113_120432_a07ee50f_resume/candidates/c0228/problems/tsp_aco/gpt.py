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

    safe = np.minimum(
        np.where(valid, distances, np.inf),
        np.where(valid.T, distances.T, np.inf),
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
    nearest = np.partition(
        safe,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    local_scale = np.mean(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(-np.clip(normalized, 0.0, 50.0))

    row_kernel = np.exp(
        -np.clip(safe / local_scale[:, None], 0.0, 50.0)
    )
    col_kernel = np.exp(
        -np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )
    endpoint_affinity = np.sqrt(row_kernel * col_kernel)

    rank_window = min(8, n - 1)
    candidates = np.argpartition(
        safe,
        kth=rank_window - 1,
        axis=1,
    )[:, :rank_window]

    candidate_distances = np.take_along_axis(
        safe,
        candidates,
        axis=1,
    )
    order = np.argsort(
        candidate_distances,
        axis=1,
        kind="stable",
    )
    neighbors = np.take_along_axis(candidates, order, axis=1)

    ranks = np.full((n, n), float(n), dtype=float)
    rank_values = np.broadcast_to(
        np.arange(rank_window, dtype=float),
        (n, rank_window),
    )
    np.put_along_axis(
        ranks,
        neighbors,
        rank_values,
        axis=1,
    )

    reciprocal_rank = 1.0 / (1.0 + ranks)

    nearest_neighbor = neighbors[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        valid
        & (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    ).astype(float)

    top_k = min(2, n - 1)
    mutual_top = (
        valid
        & (ranks < top_k)
        & (ranks.T < top_k)
    ).astype(float)

    isolation_ratio = pair_scale / (safe + global_scale)
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
