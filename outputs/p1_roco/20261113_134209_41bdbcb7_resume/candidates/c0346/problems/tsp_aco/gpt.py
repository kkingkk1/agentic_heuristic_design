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
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    safe = np.where(finite, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, np.finfo(float).eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    available = np.isfinite(
        np.take_along_axis(safe, order, axis=1)
    )

    max_k = max(1, min(n - 1, 6))
    candidate_distances = np.take_along_axis(
        safe,
        order[:, :max_k],
        axis=1,
    )
    candidate_valid = np.isfinite(candidate_distances)

    counts = candidate_valid.sum(axis=1)
    local_scale = np.divide(
        np.where(
            candidate_valid,
            candidate_distances,
            0.0,
        ).sum(axis=1),
        np.maximum(counts, 1),
    )
    local_scale = np.where(
        counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = safe / np.maximum(pair_scale, np.finfo(float).eps)

    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 40.0)
    )

    endpoint_score = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 40.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 40.0))
    )

    rank_score = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    rank_score = 0.35 + 0.65 * rank_score

    adaptive_k = max(2, min(n - 1, int(round(np.sqrt(n)))))
    top_k = min(max_k, adaptive_k)

    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True

    top_neighbors &= valid

    reciprocal = top_neighbors & top_neighbors.T
    reciprocal_score = 1.0 + 0.16 * reciprocal.astype(float)

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    neighborhood_sizes = top_neighbors.sum(axis=1).astype(float)
    union_sizes = (
        neighborhood_sizes[:, None]
        + neighborhood_sizes[None, :]
        - shared_neighbors
    )

    overlap = shared_neighbors / np.maximum(union_sizes, 1.0)

    competitive = (
        valid
        & (ranks < top_k)
        & (ranks.T < top_k)
    )

    complementary_score = (
        1.0
        + 0.14
        * (1.0 - np.clip(overlap, 0.0, 1.0))
        * competitive.astype(float)
    )

    two_hop_score = (
        1.0
        + 0.04
        * np.clip(
            shared_neighbors / max(float(top_k), 1.0),
            0.0,
            1.0,
        )
        * competitive.astype(float)
    )

    candidate_popularity = top_neighbors.astype(float).sum(axis=0)
    popularity_baseline = max(
        float(np.mean(candidate_popularity)),
        1.0,
    )
    popularity_ratio = np.clip(
        candidate_popularity / popularity_baseline,
        0.0,
        3.0,
    )

    popularity_score = 1.0 / np.sqrt(
        (1.0 + 0.11 * popularity_ratio[:, None])
        * (1.0 + 0.11 * popularity_ratio[None, :])
    )

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(
        endpoint_best,
        np.finfo(float).eps,
    )

    opportunity_score = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    isolation_score = 1.0 + 0.08 * np.clip(
        pair_scale / np.maximum(safe + global_scale, np.finfo(float).eps),
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_score
        * rank_score
        * reciprocal_score
        * complementary_score
        * two_hop_score
        * popularity_score
        * opportunity_score
        * isolation_score
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
