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
    valid_input = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    masked = np.where(valid, safe, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")

    usable_k = min(3, n - 1)
    nearest_idx = order[:, :usable_k]
    nearest_dist = np.take_along_axis(masked, nearest_idx, axis=1)
    nearest_valid = np.isfinite(nearest_dist)
    counts = np.sum(nearest_valid, axis=1)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_dist, 0.0),
        axis=1,
    ) / np.maximum(counts, 1)

    local_scale = np.where(
        counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    # Use a mildly robust pair scale to avoid over-penalizing one heterogeneous endpoint.
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    normalized_distance = safe / pair_scale
    distance_log = -np.clip(normalized_distance, 0.0, 50.0)

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

    # Smooth rank signal; it remains useful when distance scales are uneven.
    rank_signal = -0.5 * (
        np.log1p(row_rank) + np.log1p(column_rank)
    )

    endpoint_affinity = -0.5 * (
        np.clip(safe / local_scale[:, None], 0.0, 50.0)
        + np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )

    # Penalize only edges clearly worse than both endpoints' local alternatives.
    alternative_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    regret = np.maximum(
        safe / np.maximum(alternative_scale, eps) - 1.0,
        0.0,
    )
    regret_penalty = -0.16 * np.clip(regret, 0.0, 4.0)

    top_k = min(2, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True

    mutual_top = (valid & top_neighbors & top_neighbors.T).astype(float)

    nearest_row = order[:, 0]
    nearest_column = column_order[0, :]
    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_column[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Small additive bonuses avoid the excessive concentration of multiplicative rewards.
    mutual_signal = 0.12 * mutual_top + 0.18 * mutual_nearest

    # Weighted log fusion provides conservative calibration and preserves diversity.
    log_scores = (
        0.43 * distance_log
        + 0.22 * rank_signal
        + 0.23 * endpoint_affinity
        + regret_penalty
        + mutual_signal
    )

    log_scores[~valid] = -np.inf
    finite_scores = np.where(np.isfinite(log_scores), log_scores, 0.0)

    # Shift only valid entries before exponentiation for numerical stability.
    valid_max = np.max(finite_scores[valid])
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
