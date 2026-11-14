import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(
        valid,
        0.5 * (distances + distances.T),
        np.inf,
    )
    np.fill_diagonal(safe, np.inf)

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

    order = np.argsort(safe, axis=1, kind="stable")
    rank = np.full((n, n), float(n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank[~valid] = float(n)

    valid_counts = np.sum(valid, axis=1)
    k = min(4, n - 1)

    nearest_order = order[:, :k]
    nearest_distances = np.take_along_axis(safe, nearest_order, axis=1)
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

    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    rank_score = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )
    rank_factor = 0.88 + 0.12 * rank_score

    top_neighbors = valid & (rank < float(k))
    mutual_top_neighbors = top_neighbors & top_neighbors.T

    base_scores = (
        distance_score
        * rank_factor
        * (1.0 + 0.12 * mutual_top_neighbors.astype(float))
    )
    base_scores[~valid] = 0.0

    # Conservative long-horizon signal: reward edges that remain candidates
    # under a slightly wider neighborhood in both directions.
    long_k = min(5, n - 1)
    long_neighbors = valid & (rank < float(long_k))
    persistent = (
        top_neighbors
        & long_neighbors
        & top_neighbors.T
        & long_neighbors.T
    )

    long_strength = np.zeros((n, n), dtype=float)
    long_strength[long_neighbors] = np.exp(
        -np.clip(
            safe[long_neighbors]
            / np.maximum(
                local_scale[:, None].repeat(n, axis=1)[long_neighbors],
                eps,
            ),
            0.0,
            50.0,
        )
    )

    sorted_strength = np.sort(long_strength, axis=1)
    strongest = sorted_strength[:, -1]
    second = (
        sorted_strength[:, -2]
        if long_k >= 2
        else np.zeros(n, dtype=float)
    )

    reserve = np.clip(
        second / np.maximum(strongest, eps),
        0.0,
        1.0,
    )
    endpoint_reserve = np.sqrt(
        reserve[:, None] * reserve[None, :]
    )

    # Keep the exploratory correction small and bounded.
    reserve_signal = endpoint_reserve * persistent.astype(float)
    scores = base_scores * np.exp(0.035 * reserve_signal)
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
