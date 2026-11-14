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

    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    # Treat an edge as available if either direction is finite.
    safe = np.where(finite_nonnegative, distances, np.inf)
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

    usable_k = min(3, n - 1)
    nearest_idx = order[:, :usable_k]
    nearest_dist = np.take_along_axis(masked, nearest_idx, axis=1)
    nearest_valid = np.isfinite(nearest_dist)
    neighbor_count = np.sum(nearest_valid, axis=1)

    # Robust local scale: trim the largest of three neighbors when possible.
    sorted_nearest = np.sort(
        np.where(nearest_valid, nearest_dist, np.inf),
        axis=1,
    )
    if usable_k >= 3:
        scale_values = sorted_nearest[:, :2]
        scale_valid = np.isfinite(scale_values)
    else:
        scale_values = sorted_nearest
        scale_valid = np.isfinite(scale_values)

    scale_sum = np.sum(
        np.where(scale_valid, scale_values, 0.0),
        axis=1,
    )
    scale_count = np.sum(scale_valid, axis=1)

    local_scale = scale_sum / np.maximum(scale_count, 1)
    local_scale = np.where(
        scale_count > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    # Zero-distance edges should remain highly preferred without causing division issues.
    positive_local = local_scale[local_scale > eps]
    fallback_scale = (
        float(np.median(positive_local))
        if positive_local.size
        else global_scale
    )
    denominator = np.maximum(local_scale, fallback_scale * 1e-6)
    denominator = np.maximum(denominator, eps)

    pair_scale = np.sqrt(denominator[:, None] * denominator[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    normalized_distance = np.divide(
        safe,
        pair_scale,
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )

    # Strong but bounded short-edge preference.
    distance_signal = -np.log1p(np.clip(normalized_distance, 0.0, 1e6))

    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    column_order = np.argsort(masked, axis=0, kind="stable")
    column_rank = np.empty((n, n), dtype=float)
    column_rank[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    # Relative rank helps when edge lengths have heterogeneous scales.
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

    # Bounded opportunity-cost penalty for edges clearly worse than both endpoints' alternatives.
    alternative_scale = np.minimum(
        denominator[:, None],
        denominator[None, :],
    )
    relative_regret = np.divide(
        safe,
        np.maximum(alternative_scale, eps),
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    ) - 1.0
    regret_penalty = -0.14 * np.clip(relative_regret, 0.0, 4.0)

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
    ).astype(float)

    nearest_row = order[:, 0]
    nearest_column = column_order[0, :]
    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_column[None, :] == np.arange(n)[:, None])
    ).astype(float)

    mutual_signal = 0.10 * mutual_top + 0.15 * mutual_nearest

    # Sparse-node protection prevents necessary edges from being suppressed.
    degree = np.sum(valid, axis=1)
    sparse_endpoint = (
        (degree[:, None] <= top_k)
        | (degree[None, :] <= top_k)
    ).astype(float)
    sparse_protection = 0.10 * sparse_endpoint * np.exp(
        -np.clip(normalized_distance, 0.0, 12.0)
    )

    log_scores = (
        0.48 * distance_signal
        + 0.21 * rank_signal
        + 0.21 * endpoint_affinity
        + regret_penalty
        + mutual_signal
        + sparse_protection
    )

    log_scores[~valid] = -np.inf

    finite_scores = np.where(np.isfinite(log_scores), log_scores, 0.0)
    valid_max = float(np.max(finite_scores[valid]))

    scores = np.zeros((n, n), dtype=float)
    scores[valid] = np.exp(
        np.clip(finite_scores[valid] - valid_max, -50.0, 0.0)
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
