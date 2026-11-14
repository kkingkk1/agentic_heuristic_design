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

    safe = np.where(
        valid | valid.T,
        np.minimum(
            np.where(valid, distances, np.inf),
            np.where(valid.T, distances.T, np.inf),
        ),
        np.inf,
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

    local_k = min(4, n - 1)
    nearest_values = np.partition(
        safe,
        local_k - 1,
        axis=1,
    )[:, :local_k]

    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = safe / pair_scale
    distance_kernel = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    endpoint_kernel = np.exp(
        -0.5
        * np.clip(
            safe / local_scale[:, None]
            + safe / local_scale[None, :],
            0.0,
            100.0,
        )
    )

    rank_k = min(8, n - 1)
    candidate_ids = np.argpartition(
        safe,
        rank_k - 1,
        axis=1,
    )[:, :rank_k]

    candidate_distances = np.take_along_axis(
        safe,
        candidate_ids,
        axis=1,
    )
    order = np.argsort(
        candidate_distances,
        axis=1,
        kind="stable",
    )
    neighbors = np.take_along_axis(
        candidate_ids,
        order,
        axis=1,
    )

    ranks = np.full((n, n), float(n), dtype=float)
    rank_values = np.arange(rank_k, dtype=float)
    np.put_along_axis(
        ranks,
        neighbors,
        rank_values[None, :],
        axis=1,
    )

    reciprocal_rank = 1.0 / (1.0 + ranks)

    nearest_neighbor = neighbors[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        valid
        & (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    )

    mutual_top = (
        valid
        & (ranks < min(2, n - 1))
        & (ranks.T < min(2, n - 1))
    )

    isolation_ratio = pair_scale / (safe + global_scale)
    isolation_bonus = 1.0 + 0.20 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    scores = (
        distance_kernel
        * endpoint_kernel
        * (0.50 + 0.50 * reciprocal_rank)
        * (
            1.0
            + 0.30 * mutual_nearest.astype(float)
            + 0.10 * mutual_top.astype(float)
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
