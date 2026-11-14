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

    safe = np.where(raw_valid, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    valid_counts = valid.sum(axis=1)

    usable = min(4, n - 1)
    nearest = np.take_along_axis(safe, order[:, :usable], axis=1)
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
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    distance_score = np.exp(
        -np.clip(safe / np.maximum(pair_scale, eps), 0.0, 40.0)
    )

    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order[:, :usable],
    ] = np.arange(usable, dtype=float)[None, :]
    row_rank = np.where(valid, row_rank, float(n))

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )
    rank_factor = 0.72 + 0.28 * reciprocal_rank

    nearest_neighbor = np.full(n, -1, dtype=int)
    has_neighbor = valid_counts > 0
    if np.any(has_neighbor):
        nearest_neighbor[has_neighbor] = order[has_neighbor, 0]

    indices = np.arange(n)
    mutual_nearest = (
        valid
        & (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    ).astype(float)

    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    if top_k > 0:
        top_neighbors[
            np.arange(n)[:, None],
            order[:, :top_k],
        ] = valid[
            np.arange(n)[:, None],
            order[:, :top_k],
        ]

    mutual_top = (
        valid
        & top_neighbors
        & top_neighbors.T
    ).astype(float)

    mutual_bonus = (
        1.0
        + 0.18 * mutual_nearest
        + 0.06 * mutual_top
    )

    competitive = (
        valid
        & (row_rank < min(4, n - 1))
        & (row_rank.T < min(4, n - 1))
    )

    endpoint_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_scale, eps)
    opportunity_penalty = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 3.0)
    )

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )
    union_neighbors = (
        top_neighbors.sum(axis=1)[:, None]
        + top_neighbors.sum(axis=1)[None, :]
        - shared_neighbors
    )
    overlap = shared_neighbors / np.maximum(union_neighbors, 1.0)

    complementary_factor = (
        1.0
        + 0.06
        * np.clip(1.0 - overlap, 0.0, 1.0)
        * competitive.astype(float)
    )

    scores = (
        distance_score
        * rank_factor
        * mutual_bonus
        * opportunity_penalty
        * complementary_factor
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
