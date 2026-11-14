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

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    # Use the best available directed value so the resulting priorities are symmetric.
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

    # Exclude invalid entries from local-neighborhood statistics.
    usable_k = min(3, n - 1)
    masked = np.where(valid, safe, np.inf)
    row_order = np.argsort(masked, axis=1, kind="stable")

    nearest_indices = row_order[:, :usable_k]
    nearest_distances = np.take_along_axis(
        masked,
        nearest_indices,
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_distances)
    counts = np.sum(nearest_valid, axis=1)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_distances, 0.0),
        axis=1,
    ) / np.maximum(counts, 1)

    local_scale = np.where(
        counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    # Stable rank preference, averaged across both endpoints.
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(masked, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    # Reward only mutual nearest and mutual top-two edges.
    nearest_row = row_order[:, 0]
    nearest_col = col_order[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_k = min(2, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    mutual_top_two = (
        valid
        & top_neighbors
        & top_neighbors.T
    ).astype(float)

    # Bounded opportunity cost against each endpoint's local alternatives.
    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_best, eps)

    opportunity_penalty = 1.0 / (
        1.0
        + 0.20 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    # Mild two-sided local competitiveness gate, without shared-neighbor rewards.
    competitive_k = min(3, n - 1)
    locally_competitive = (
        valid
        & (row_rank < competitive_k)
        & (col_rank < competitive_k)
    ).astype(float)

    mutual_bonus = (
        1.0
        + 0.24 * mutual_nearest
        + 0.08 * mutual_top_two
    )

    local_competition_bonus = (
        1.0
        + 0.04 * locally_competitive
    )

    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 50.0))
    )

    scores = (
        distance_score
        * endpoint_affinity
        * (0.62 + 0.38 * reciprocal_rank)
        * opportunity_penalty
        * mutual_bonus
        * local_competition_bonus
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
