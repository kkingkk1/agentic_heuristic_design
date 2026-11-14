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
    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    # Symmetrize directed input conservatively: retain the shorter available
    # direction while rejecting edges unavailable in both directions.
    forward = np.where(finite_nonnegative, distances, np.inf)
    backward = np.where(finite_nonnegative.T, distances.T, np.inf)
    edge_distance = np.minimum(forward, backward)
    valid = np.isfinite(edge_distance) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    valid_values = edge_distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(valid_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Stable ranks place finite edges before unavailable edges.
    order = np.argsort(edge_distance, axis=1, kind="stable")
    rank = np.full((n, n), float(n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    degree = valid.sum(axis=1)
    local_k = min(5, n - 1)

    nearest = np.take_along_axis(edge_distance, order[:, :local_k], axis=1)
    nearest_valid = np.isfinite(nearest)

    local_scale = np.divide(
        np.where(nearest_valid, nearest, 0.0).sum(axis=1),
        nearest_valid.sum(axis=1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_valid.sum(axis=1) > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = edge_distance / np.maximum(pair_scale, eps)

    # Smooth distance preference, robust to very large distances.
    distance_term = np.exp(-np.clip(relative_distance, 0.0, 40.0))
    global_term = np.exp(
        -np.clip(edge_distance / global_scale, 0.0, 40.0)
    )

    # Reciprocal rank rewards edges that are attractive from both endpoints.
    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )
    rank_term = 0.35 + 0.65 * reciprocal_rank

    top_k = min(3, n - 1)
    top = valid & (rank < float(top_k))

    mutual_top = top & top.T
    mutual_nearest = (
        valid
        & (rank == 0.0)
        & (rank.T == 0.0)
    )

    structure_term = (
        1.0
        + 0.16 * mutual_top.astype(float)
        + 0.24 * mutual_nearest.astype(float)
    )

    # Penalize edges that are expensive relative to at least one endpoint's
    # best alternatives, while retaining edges needed by sparse endpoints.
    best_alternative = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    regret_ratio = edge_distance / np.maximum(best_alternative, eps)
    regret_term = 1.0 / (
        1.0
        + 0.10 * np.clip(regret_ratio - 1.0, 0.0, 6.0)
    )

    # Shared neighborhoods indicate local geometric compatibility.
    top_float = top.astype(float)
    shared = top_float @ top_float.T
    neighborhood_term = 1.0 + 0.08 * np.clip(
        shared / max(float(top_k), 1.0),
        0.0,
        1.0,
    )

    # Avoid assigning too much probability to edges incident to globally
    # popular candidate nodes.
    popularity = top_float.sum(axis=0)
    popularity_reference = max(float(np.mean(popularity)), 1.0)
    popularity_ratio = popularity / popularity_reference
    popularity_term = 1.0 / np.sqrt(
        (1.0 + 0.08 * np.clip(popularity_ratio[:, None], 0.0, 4.0))
        * (1.0 + 0.08 * np.clip(popularity_ratio[None, :], 0.0, 4.0))
    )

    # Sparse endpoints receive a small compensation so that their only
    # feasible connections are not suppressed by global popularity effects.
    density = degree / max(float(n - 1), 1.0)
    sparsity = 1.0 - density
    sparsity_term = 1.0 + 0.10 * np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )

    scores = (
        distance_term
        * (0.35 + 0.65 * global_term)
        * rank_term
        * structure_term
        * regret_term
        * neighborhood_term
        * popularity_term
        * sparsity_term
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
