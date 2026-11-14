import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)

    # Require validity in both directions before constructing an undirected edge.
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distances = 0.5 * (distances + distances.T)
    safe = np.where(valid, symmetric_distances, np.inf)

    valid_values = safe[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif valid_values.size:
        global_scale = max(float(np.median(valid_values)), 1.0)
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    # Stable ordering gives deterministic behavior under tied distances.
    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)

    rank_values = np.arange(n, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = rank_values[None, :]
    row_rank[~valid] = float(n)

    valid_counts = np.sum(valid, axis=1)
    neighborhood_size = min(4, n - 1)

    nearest_distances = np.take_along_axis(
        safe,
        row_order[:, :neighborhood_size],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_distances)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_distances, 0.0),
        axis=1,
    ) / np.maximum(np.sum(nearest_valid, axis=1), 1)

    local_scale = np.where(
        valid_counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    # Distance remains the dominant signal.
    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    # Mild rank decay reduces the influence of distant alternatives.
    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )
    rank_factor = 0.88 + 0.12 * rank_score

    # A single conservative bidirectional-neighborhood bonus.
    top_k = min(4, n - 1)
    top_neighbors = valid & (row_rank < float(top_k))
    mutual_top_neighbors = top_neighbors & top_neighbors.T
    mutual_factor = 1.0 + 0.12 * mutual_top_neighbors.astype(float)

    scores = distance_score * rank_factor * mutual_factor
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
