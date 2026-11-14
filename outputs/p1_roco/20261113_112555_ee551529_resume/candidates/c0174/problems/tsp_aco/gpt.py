import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric distance- and neighborhood-aware TSP edge prior."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances)
    valid_directed = finite & (distances >= 0.0) & ~diagonal

    # TSP edge priors should only use edges valid in both directions.
    valid = valid_directed & valid_directed.T
    if not np.any(valid):
        return result

    # Symmetrize distances while preserving invalid edges.
    edge_distance = np.where(valid, 0.5 * (distances + distances.T), np.inf)

    valid_values = edge_distance[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(valid_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Per-node local scale from the nearest usable neighbors.
    neighbor_count = np.sum(valid, axis=1)
    k = min(3, max(1, n - 1))

    sorted_distances = np.sort(edge_distance, axis=1)
    nearest = sorted_distances[:, :k]
    local_scale = np.sum(
        np.where(np.isfinite(nearest), nearest, 0.0),
        axis=1,
    ) / np.maximum(
        np.sum(np.isfinite(nearest), axis=1),
        1,
    )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    # Robust shortness relative to both endpoint neighborhoods.
    normalized_distance = np.divide(
        edge_distance,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    shortness = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    # Reciprocal ranks computed only over valid neighbors.
    rank_order = np.argsort(edge_distance, axis=1, kind="stable")
    ranks = np.full((n, n), np.inf, dtype=float)

    row_index = np.arange(n)[:, None]
    rank_positions = np.arange(n, dtype=float)[None, :]
    ordered_valid = valid[row_index, rank_order]

    ranks[row_index, rank_order] = np.where(
        ordered_valid,
        rank_positions,
        np.inf,
    )

    reciprocal_rank = np.zeros((n, n), dtype=float)
    reciprocal_rank[valid] = 1.0 / np.sqrt(
        (1.0 + ranks[valid])
        * (1.0 + ranks.T[valid])
    )

    # Reinforce mutual nearest neighbors.
    nearest_index = np.argmin(edge_distance, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_index[i]
        if valid[i, j] and nearest_index[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Mildly favor edges useful to locally isolated endpoints.
    isolation = np.divide(
        pair_scale,
        edge_distance + global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    isolation = np.clip(isolation, 0.0, 1.0)

    # Bridge term rewards edges incident to nodes whose local scale is large.
    node_isolation = np.clip(local_scale / global_scale, 0.0, 4.0)
    bridge_bonus = 1.0 + 0.08 * np.minimum(
        node_isolation[:, None] + node_isolation[None, :],
        4.0,
    )

    score = (
        shortness
        * (0.55 + 0.45 * reciprocal_rank)
        * (1.0 + 0.25 * mutual)
        * (1.0 + 0.15 * isolation)
        * bridge_bonus
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
