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
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    usable = min(5, n - 1)
    nearest = np.take_along_axis(safe, row_order[:, :usable], axis=1)
    nearest_valid = np.isfinite(nearest)

    local_scale = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    ) / np.maximum(nearest_valid.sum(axis=1), 1)

    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 40.0))

    endpoint_score = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 40.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 40.0))
    )

    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )
    rank_score = 0.55 + 0.45 * rank_score

    top_k = min(4, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True
    top_neighbors &= valid

    reciprocal = top_neighbors & top_neighbors.T

    reciprocal_rank = np.sqrt(
        (1.0 / (1.0 + row_rank))
        * (1.0 / (1.0 + row_rank.T))
    )

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    degree = top_neighbors.sum(axis=1).astype(float)
    union = degree[:, None] + degree[None, :] - shared_neighbors
    overlap = shared_neighbors / np.maximum(union, 1.0)

    competitive_k = min(5, n - 1)
    competitive = (
        valid
        & (row_rank < competitive_k)
        & (row_rank.T < competitive_k)
    )

    reciprocal_bonus = (
        1.0
        + 0.12 * reciprocal.astype(float)
        + 0.05 * reciprocal_rank * competitive.astype(float)
    )

    structural_complement = (
        1.0
        + 0.12
        * np.clip(1.0 - overlap, 0.0, 1.0)
        * competitive.astype(float)
    )

    two_hop_support = np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
    )
    two_hop_bonus = (
        1.0
        + 0.035 * two_hop_support * competitive.astype(float)
    )

    candidate_popularity = top_neighbors.astype(float).sum(axis=0)
    popularity_reference = max(float(np.mean(candidate_popularity)), 1.0)
    popularity_ratio = np.clip(
        candidate_popularity / popularity_reference,
        0.0,
        3.0,
    )

    popularity_penalty = 1.0 / np.sqrt(
        (1.0 + 0.12 * popularity_ratio[:, None])
        * (1.0 + 0.12 * popularity_ratio[None, :])
    )

    local_competition = np.sum(
        np.exp(-np.clip(safe / np.maximum(local_scale[:, None], eps), 0.0, 30.0)),
        axis=1,
    )
    local_competition = np.where(
        np.isfinite(local_competition),
        local_competition,
        0.0,
    )

    scarcity = 1.0 / np.sqrt(1.0 + local_competition)
    scarcity_reference = max(float(np.median(scarcity)), eps)
    scarcity_ratio = np.clip(
        scarcity / scarcity_reference,
        0.5,
        2.0,
    )

    isolation_bonus = np.sqrt(
        scarcity_ratio[:, None] * scarcity_ratio[None, :]
    )
    isolation_bonus = 1.0 + 0.08 * np.clip(isolation_bonus - 1.0, -0.5, 1.0)

    opportunity_ratio = safe / np.maximum(
        np.minimum(local_scale[:, None], local_scale[None, :]),
        eps,
    )
    opportunity_penalty = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    scores = (
        distance_score
        * endpoint_score
        * rank_score
        * reciprocal_bonus
        * structural_complement
        * two_hop_bonus
        * popularity_penalty
        * isolation_bonus
        * opportunity_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
