import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promising indicators for a TSP distance matrix."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)

    positive_edges = d[np.isfinite(d) & (d > 0)]
    if positive_edges.size == 0:
        return np.zeros_like(d, dtype=float)

    global_scale = float(np.median(positive_edges))
    global_scale = max(global_scale, np.finfo(float).eps)

    # Each node's local scale is based on its nearest non-zero incident edges.
    sorted_distances = np.sort(d, axis=1)
    k = max(1, min(n - 1, int(np.sqrt(n))))
    local_scale = np.median(sorted_distances[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    # Rank of each edge from the perspective of each endpoint.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values
    ranks = np.minimum(ranks, float(n - 1))
    np.fill_diagonal(ranks, np.inf)

    i_scale = local_scale[:, None]
    j_scale = local_scale[None, :]

    # Local distance compatibility from both endpoints.
    local_compatibility = np.exp(-d / (i_scale + j_scale + global_scale * 1e-12))

    # Mutual rank favors edges that are independently attractive at both ends.
    mutual_rank = 1.0 / (ranks + ranks.T)

    # Global scale term prevents extremely long edges from receiving high scores.
    global_compatibility = np.exp(-d / global_scale)

    # Mild preference for edges whose endpoint scales are compatible.
    scale_balance = np.exp(
        -np.abs(np.log((i_scale + 1e-12) / (j_scale + 1e-12)))
    )

    score = (
        local_compatibility
        * (0.5 + mutual_rank)
        * (0.5 + global_compatibility)
        * (0.5 + scale_balance)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
