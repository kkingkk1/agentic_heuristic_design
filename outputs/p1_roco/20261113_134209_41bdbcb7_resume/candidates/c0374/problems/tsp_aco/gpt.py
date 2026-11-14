import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, distance-dominant edge-priority matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    directed_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(directed_valid):
        return scores

    safe = np.minimum(
        np.where(directed_valid, distances, np.inf),
        np.where(directed_valid.T, distances.T, np.inf),
    )
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

    valid_counts = valid.sum(axis=1)
    neighbor_count = min(4, n - 1)

    row_order = np.argsort(safe, axis=1, kind="stable")
    nearest_indices = row_order[:, :neighbor_count]
    nearest_distances = np.take_along_axis(safe, nearest_indices, axis=1)
    nearest_valid = np.isfinite(nearest_distances)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_distances, 0.0),
        axis=1,
    ) / np.maximum(nearest_valid.sum(axis=1), 1)

    local_scale = np.where(valid_counts > 0, local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 50.0))
    distance_score[~valid] = 0.0

    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )
    rank_score[~valid] = 0.0

    nearest_neighbor = np.full(n, -1, dtype=int)
    for i in range(n):
        valid_neighbors = np.flatnonzero(valid[i])
        if valid_neighbors.size:
            nearest_neighbor[i] = valid_neighbors[
                np.argmin(safe[i, valid_neighbors])
            ]

    indices = np.arange(n)
    mutual_nearest = (
        valid
        & (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    ).astype(float)

    top_k = min(3, n - 1)
    mutual_top = (
        valid
        & (row_rank < float(top_k))
        & (row_rank.T < float(top_k))
    ).astype(float)

    # Distance remains dominant; rank supplies a bounded exploration floor.
    scores = (
        0.78 * distance_score
        + 0.22 * rank_score
    )

    # Small, capped local bonuses preserve mutual-neighbor evidence
    # without overwhelming the distance signal.
    scores *= (
        1.0
        + 0.16 * mutual_nearest
        + 0.05 * mutual_top
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
