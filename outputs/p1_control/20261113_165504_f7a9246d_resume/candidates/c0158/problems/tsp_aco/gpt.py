import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric stochastic edge-priority indicators for a TSP matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must be nonnegative")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    # Use a symmetric metric while retaining the original scale.
    d = 0.5 * (distances + distances.T)
    valid = ~np.eye(n, dtype=bool)
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    positive = work[valid & (work > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_d = np.take_along_axis(work, order, axis=1)

    # Robust endpoint-local scale from the closest few neighbors.
    k = min(4, n - 1)
    neighborhood = sorted_d[:, :k]
    positive_neighborhood = neighborhood[
        np.isfinite(neighborhood) & (neighborhood > 0)
    ]

    if positive_neighborhood.size:
        local_fallback = float(np.median(positive_neighborhood))
    else:
        local_fallback = global_scale

    local_scale = np.nanmedian(
        np.where(np.isfinite(neighborhood), neighborhood, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        local_fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    # Global and endpoint-relative distance evidence.
    global_affinity = np.exp(-np.minimum(work / global_scale, 50.0))
    local_affinity = np.exp(-np.minimum(work / pair_scale, 50.0))

    # Reciprocal endpoint rank: an edge is strong when both endpoints rank it well.
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_i = ranks
    rank_j = ranks.T
    rank_affinity = np.exp(
        -0.5 * (rank_i + rank_j) / max(1.5, np.sqrt(float(n)))
    )

    # Soft candidate support preserves a weak global signal outside the shortlist.
    candidate_limit = min(4, n - 1)
    candidate_support = np.exp(
        -0.55 * np.maximum(np.minimum(rank_i, rank_j) - candidate_limit, 0.0)
    )

    # Reward edges that are competitive for both endpoints.
    nearest = sorted_d[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    competitiveness = np.exp(
        -np.minimum(
            0.5 * (
                work / np.maximum(nearest[:, None], eps)
                + work / np.maximum(nearest[None, :], eps)
            ),
            50.0,
        )
    )

    # Regret identifies vertices for which losing the best local edge is costly.
    if n >= 3:
        second = sorted_d[:, 1]
        second = np.where(np.isfinite(second), second, nearest)
        regret = np.clip(
            (second - nearest) / np.maximum(second, global_scale * 1e-12),
            0.0,
            1.0,
        )
    else:
        regret = np.zeros(n, dtype=float)

    nearest_neighbor = order[:, 0]
    mutual_nearest = (
        nearest_neighbor[nearest_neighbor] == np.arange(n)
    ).astype(float)
    mutual_bonus = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    cols = nearest_neighbor
    mutual_bonus[rows, cols] = mutual_nearest
    mutual_bonus[cols, rows] = mutual_nearest

    regret_bonus = 0.5 * (
        regret[:, None] * (rank_i == 0)
        + regret[None, :] * (rank_j == 0)
    )

    score = (
        0.24 * global_affinity
        + 0.27 * local_affinity
        + 0.20 * rank_affinity
        + 0.17 * competitiveness
        + 0.07 * regret_bonus
        + 0.05 * mutual_bonus
    )

    score *= 0.82 + 0.18 * candidate_support
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    mean_score = float(np.mean(score[valid]))
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score.astype(float, copy=False)
