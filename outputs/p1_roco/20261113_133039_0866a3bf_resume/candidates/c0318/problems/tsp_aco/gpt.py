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
    raw_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(raw_valid):
        return np.zeros((n, n), dtype=float)

    # Use the better direction while enforcing an undirected TSP edge model.
    safe = np.where(raw_valid, distances, np.inf)
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

    order = np.argsort(safe, axis=1, kind="stable")
    rank = np.full((n, n), float(n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    valid_counts = np.sum(valid, axis=1)
    small_k = min(2, n - 1)
    large_k = min(5, n - 1)

    small_neighbors = (
        valid & (rank < float(small_k))
    ).astype(float)
    large_neighbors = (
        valid & (rank < float(large_k))
    ).astype(float)

    # Neighborhood scales are computed at two resolutions and then blended.
    small_values = np.take_along_axis(
        safe,
        order[:, :small_k],
        axis=1,
    )
    large_values = np.take_along_axis(
        safe,
        order[:, :large_k],
        axis=1,
    )

    small_finite = np.isfinite(small_values)
    large_finite = np.isfinite(large_values)

    small_scale = (
        np.sum(np.where(small_finite, small_values, 0.0), axis=1)
        / np.maximum(np.sum(small_finite, axis=1), 1)
    )
    large_scale = (
        np.sum(np.where(large_finite, large_values, 0.0), axis=1)
        / np.maximum(np.sum(large_finite, axis=1), 1)
    )

    small_scale = np.where(valid_counts > 0, small_scale, global_scale)
    large_scale = np.where(valid_counts > 0, large_scale, global_scale)
    small_scale = np.maximum(small_scale, eps)
    large_scale = np.maximum(large_scale, eps)

    pair_small = np.sqrt(small_scale[:, None] * small_scale[None, :])
    pair_large = np.sqrt(large_scale[:, None] * large_scale[None, :])

    # Distance remains the dominant signal, with local scale adaptation.
    local_distance_score = np.exp(
        -np.clip(safe / np.maximum(pair_small, eps), 0.0, 50.0)
    )
    broad_distance_score = np.exp(
        -np.clip(safe / np.maximum(pair_large, eps), 0.0, 50.0)
    )
    distance_score = np.sqrt(local_distance_score * broad_distance_score)

    # Smooth rank preference preserves ordering without making rank dominant.
    rank_score = 1.0 / np.sqrt((1.0 + rank) * (1.0 + rank.T))

    nearest = order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    mutual_small = (
        small_neighbors
        * small_neighbors.T
    )
    mutual_large = (
        large_neighbors
        * large_neighbors.T
    )

    # Shared neighborhoods provide a bounded multi-hop completion signal.
    shared_small = (
        small_neighbors @ small_neighbors.T
    ) / max(float(small_k), 1.0)
    shared_large = (
        large_neighbors @ large_neighbors.T
    ) / max(float(large_k), 1.0)

    shared_signal = np.clip(
        0.65 * shared_small + 0.35 * shared_large,
        0.0,
        1.0,
    )

    # Mild bridge/opportunity adjustment: avoid edges that are unusually
    # expensive for either endpoint, but do not discard them outright.
    endpoint_scale = np.minimum(
        small_scale[:, None],
        small_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_scale, eps)
    opportunity_adjustment = 1.0 / (
        1.0 + 0.12 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    # Alternative-edge ratio measures complementary coverage.
    local_strength = np.where(
        valid,
        np.exp(-np.clip(safe / small_scale[:, None], 0.0, 50.0))
        / np.sqrt(1.0 + rank),
        0.0,
    )

    strongest = np.max(local_strength, axis=1)
    sorted_strength = np.sort(local_strength, axis=1)
    second = sorted_strength[:, -2] if n >= 2 else np.zeros(n)

    alternative_ratio = second / np.maximum(strongest, eps)
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)
    complementary_signal = np.sqrt(
        alternative_ratio[:, None] * alternative_ratio[None, :]
    )

    # Penalize concentration only for edges already near the top of both
    # endpoint rankings, keeping essential sparse-node edges available.
    total_strength = np.sum(local_strength, axis=1)
    concentration = strongest / np.maximum(total_strength, eps)
    concentration = np.clip(concentration, 0.0, 1.0)

    topness = np.sqrt(
        np.clip(1.0 / (1.0 + rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + rank.T), 0.0, 1.0)
    )
    concentration_penalty = (
        np.sqrt(concentration[:, None] * concentration[None, :])
        * topness
    )

    log_scores = (
        np.log(np.maximum(distance_score, eps))
        + 0.22 * np.log1p(2.0 * rank_score)
        + 0.10 * mutual_nearest
        + 0.045 * mutual_small
        + 0.025 * mutual_large
        + 0.055 * shared_signal * large_neighbors * large_neighbors.T
        + np.log(np.maximum(opportunity_adjustment, eps))
        + 0.045 * complementary_signal * large_neighbors * large_neighbors.T
        - 0.055 * concentration_penalty
    )

    scores = np.exp(np.clip(log_scores, -60.0, 30.0))
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
