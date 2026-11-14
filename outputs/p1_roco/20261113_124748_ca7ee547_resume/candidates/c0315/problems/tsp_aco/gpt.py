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
    valid_input = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    # Use the best available symmetric interpretation of the matrix.
    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = max(float(np.median(safe[valid])), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    usable = min(4, n - 1)
    nearest = np.take_along_axis(safe, order[:, :usable], axis=1)
    nearest_valid = np.isfinite(nearest)

    local_scale = np.divide(
        np.where(nearest_valid, nearest, 0.0).sum(axis=1),
        np.maximum(nearest_valid.sum(axis=1), 1),
    )
    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    # Main signal: robust distance quality, calibrated to endpoint neighborhoods.
    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    # Mild reciprocal-rank support preserves edges preferred by both endpoints.
    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    rank_score = 0.72 + 0.28 * reciprocal_rank

    short_k = min(3, n - 1)
    medium_k = min(6, n - 1)

    short_neighbors = np.zeros((n, n), dtype=bool)
    short_neighbors[
        np.arange(n)[:, None],
        order[:, :short_k],
    ] = True
    short_neighbors &= valid

    medium_neighbors = np.zeros((n, n), dtype=bool)
    medium_neighbors[
        np.arange(n)[:, None],
        order[:, :medium_k],
    ] = True
    medium_neighbors &= valid

    # Multi-scale support is additive and capped, avoiding excessive
    # multiplicative amplification of possibly noisy structural signals.
    short_mutual = short_neighbors & short_neighbors.T
    medium_mutual = medium_neighbors & medium_neighbors.T

    local_support = (
        1.0
        + 0.10 * short_mutual.astype(float)
        + 0.045 * (
            medium_mutual.astype(float) - short_mutual.astype(float)
        )
    )

    # Complementary coverage rewards locally competitive edges whose endpoint
    # neighborhoods are not redundant.
    shared = short_neighbors.astype(float) @ short_neighbors.astype(float).T
    degree = short_neighbors.sum(axis=1)
    union = degree[:, None] + degree[None, :] - shared
    overlap = shared / np.maximum(union, 1.0)

    locally_competitive = (
        valid
        & (ranks < min(4, n - 1))
        & (ranks.T < min(4, n - 1))
    )

    complementary_bonus = (
        1.0
        + 0.08
        * locally_competitive.astype(float)
        * np.clip(1.0 - overlap, 0.0, 1.0)
    )

    # A bounded opportunity adjustment helps avoid edges that are unusually
    # poor for either endpoint without discarding globally short bridges.
    endpoint_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_scale, eps)
    opportunity_adjustment = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    scores = (
        distance_score
        * rank_score
        * local_support
        * complementary_bonus
        * opportunity_adjustment
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
