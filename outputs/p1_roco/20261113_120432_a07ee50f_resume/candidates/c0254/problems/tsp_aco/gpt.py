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
    valid_direction = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    # Use the best available direction while keeping the result symmetric.
    safe = np.where(valid_direction, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    valid_values = safe[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(valid_values)), 1.0)
    global_scale = max(global_scale, eps)

    masked = np.where(valid, safe, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(masked, order, axis=1)
    available = np.isfinite(sorted_distances)

    degree = np.sum(available, axis=1)
    k = min(3, n - 1)

    nearest = sorted_distances[:, :k]
    nearest_valid = np.isfinite(nearest)

    # Trim the largest neighbor when three neighbors are available.
    if k >= 3:
        scale_values = nearest[:, :2]
        scale_valid = np.isfinite(scale_values)
    else:
        scale_values = nearest
        scale_valid = np.isfinite(scale_values)

    scale_sum = np.sum(np.where(scale_valid, scale_values, 0.0), axis=1)
    scale_count = np.sum(scale_valid, axis=1)
    local_scale = scale_sum / np.maximum(scale_count, 1)

    local_scale = np.where(
        scale_count > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    positive_local = local_scale[local_scale > eps]
    fallback_scale = (
        float(np.median(positive_local))
        if positive_local.size
        else global_scale
    )
    fallback_scale = max(fallback_scale, eps)

    # Prevent zero-distance nodes from making all nonzero edges numerically vanish.
    denominator = np.maximum(local_scale, fallback_scale * 1e-6)
    pair_scale = np.sqrt(
        np.maximum(denominator[:, None] * denominator[None, :], eps)
    )

    normalized_distance = np.divide(
        safe,
        pair_scale,
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )

    clipped_distance = np.clip(normalized_distance, 0.0, 1e6)
    distance_signal = -np.log1p(clipped_distance)

    # Stable double-ended rank signal.
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    column_order = np.argsort(masked, axis=0, kind="stable")
    column_rank = np.full((n, n), float(n), dtype=float)
    column_rank[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_signal = -0.5 * (
        np.log1p(row_rank) + np.log1p(column_rank)
    )
    rank_signal = np.maximum(rank_signal, -np.log1p(float(n)))

    endpoint_ratio_i = np.divide(
        safe,
        denominator[:, None],
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )
    endpoint_ratio_j = np.divide(
        safe,
        denominator[None, :],
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )

    endpoint_affinity = -0.5 * (
        np.log1p(np.clip(endpoint_ratio_i, 0.0, 1e6))
        + np.log1p(np.clip(endpoint_ratio_j, 0.0, 1e6))
    )

    # Compare each edge with the second-best local alternative.
    second_choice = np.full(n, np.inf, dtype=float)
    if n > 1:
        second_choice[:] = sorted_distances[:, min(1, n - 1)]
    second_choice = np.where(
        np.isfinite(second_choice),
        second_choice,
        np.maximum(local_scale, global_scale),
    )
    second_choice = np.maximum(second_choice, eps)

    opportunity_ratio = np.divide(
        safe,
        np.minimum(second_choice[:, None], second_choice[None, :]),
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )

    # Only penalize clearly inferior edges, with a deliberately small weight.
    opportunity_penalty = -0.09 * np.clip(opportunity_ratio - 1.25, 0.0, 3.0)

    top_k = min(2, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True

    mutual_top = (
        valid
        & top_neighbors
        & top_neighbors.T
    )

    nearest_row = order[:, 0]
    nearest_column = column_order[0, :]
    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_column[None, :] == np.arange(n)[:, None])
    )

    mutual_signal = (
        0.075 * mutual_top.astype(float)
        + 0.10 * mutual_nearest.astype(float)
    )

    # Preserve all feasible edges incident to genuinely sparse nodes.
    sparse_endpoint = (
        (degree[:, None] <= 2)
        | (degree[None, :] <= 2)
    )
    sparse_protection = (
        0.08
        * sparse_endpoint.astype(float)
        * np.exp(-np.clip(normalized_distance, 0.0, 10.0))
    )

    log_scores = (
        0.52 * distance_signal
        + 0.22 * rank_signal
        + 0.20 * endpoint_affinity
        + opportunity_penalty
        + mutual_signal
        + sparse_protection
    )
    log_scores[~valid] = -np.inf

    valid_scores = log_scores[valid]
    if valid_scores.size == 0:
        return np.zeros((n, n), dtype=float)

    max_score = float(np.max(valid_scores))
    scores = np.zeros((n, n), dtype=float)
    scores[valid] = np.exp(
        np.clip(log_scores[valid] - max_score, -50.0, 0.0)
    )

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
