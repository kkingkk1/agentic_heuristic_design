import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric regret-aware priors for valid undirected TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid_directed = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid_directed, False)

    valid = valid_directed & valid_directed.T
    if not np.any(valid):
        return result

    pair_distance = np.where(valid, 0.5 * (d + d.T), np.inf)

    positive_edges = pair_distance[valid & (pair_distance > 0.0)]
    if positive_edges.size:
        global_scale = float(np.median(positive_edges))
    else:
        finite_edges = pair_distance[valid]
        global_scale = (
            float(np.median(finite_edges))
            if finite_edges.size
            else 1.0
        )

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(pair_distance, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    safe_distance = np.where(valid, pair_distance, np.inf)
    nearest = np.min(safe_distance, axis=1)

    second_nearest = np.full(n, np.inf, dtype=float)
    if n >= 3:
        second_nearest = safe_distance[
            np.arange(n),
            order[:, 1],
        ]

    finite_nearest = nearest[np.isfinite(nearest)]
    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        global_scale,
    )

    endpoint_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    global_score = np.exp(
        -np.minimum(safe_distance / global_scale, 50.0)
    )
    local_score = np.exp(
        -np.minimum(safe_distance / endpoint_scale, 50.0)
    )

    rank_score = np.exp(-0.62 * np.minimum(ranks, n))
    rank_score = np.where(valid, rank_score, 0.0)

    candidate_limit = min(4, max(1, n - 1))
    candidate = valid & (ranks <= candidate_limit)

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    has_alternative = (
        np.isfinite(nearest)
        & np.isfinite(second_nearest)
        & (second_nearest >= nearest)
        & (second_nearest > 0.0)
    )

    regret_strength = np.zeros(n, dtype=float)
    regret_strength[has_alternative] = np.clip(
        (second_nearest[has_alternative] - nearest[has_alternative])
        / np.maximum(second_nearest[has_alternative], eps),
        0.0,
        1.0,
    )

    pair_regret = 0.5 * (
        regret_strength[:, None] + regret_strength[None, :]
    )
    nearest_regret = nearest_edge * pair_regret

    local_competition = np.exp(
        -0.45 * (ranks + ranks.T).astype(float)
    )
    local_competition = np.where(valid & (ranks <= 1), local_competition, 0.0)

    score = (
        0.27 * global_score
        + 0.35 * local_score
        + 0.20 * rank_score
        + 0.12 * nearest_regret
        + 0.17 * mutual_nearest.astype(float)
        + 0.06 * local_competition
    )

    score = np.where(candidate, score, 0.82 * score)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
