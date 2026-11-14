import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stable, structure-aware heuristic priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    directed_valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    pair_distance = np.where(
        valid,
        0.5 * (d + d.T),
        np.inf,
    )

    finite_edges = pair_distance[valid]
    positive_edges = finite_edges[finite_edges > 0.0]

    if positive_edges.size:
        global_scale = float(np.median(positive_edges))
    elif finite_edges.size:
        global_scale = max(float(np.median(finite_edges)), 1.0)
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    nearest = np.min(pair_distance, axis=1)
    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(valid, pair_distance, np.inf)
    order = np.argsort(safe_distance, axis=1, kind="stable")

    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    edge_rank = np.where(valid, ranks, n)

    second_best = np.full(n, np.inf, dtype=float)
    if n > 2:
        second_best = safe_distance[
            np.arange(n),
            order[:, 1],
        ]

    has_two_choices = (
        np.isfinite(nearest)
        & np.isfinite(second_best)
        & (second_best > 0.0)
    )

    regret = np.zeros(n, dtype=float)
    regret[has_two_choices] = np.clip(
        (second_best[has_two_choices] - nearest[has_two_choices])
        / np.maximum(second_best[has_two_choices], eps),
        0.0,
        1.0,
    )

    global_ratio = np.divide(
        pair_distance,
        global_scale,
        out=np.full_like(pair_distance, np.inf),
        where=valid,
    )
    global_score = np.exp(-np.minimum(global_ratio, 50.0))

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_ratio = np.divide(
        pair_distance,
        pair_scale,
        out=np.full_like(pair_distance, np.inf),
        where=valid,
    )
    local_score = np.exp(-np.minimum(local_ratio, 50.0))

    rank_score = np.exp(-0.72 * np.minimum(edge_rank, n))
    rank_score = np.where(valid, rank_score, 0.0)

    nearest_edge = valid & (edge_rank == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    regret_score = np.zeros_like(pair_distance, dtype=float)
    regret_score[nearest_edge] = regret[:, None][nearest_edge]
    regret_score = 0.5 * (regret_score + regret_score.T)

    candidate_limit = min(3, max(1, n - 1))
    candidate = valid & (edge_rank <= candidate_limit)

    score = (
        0.34 * global_score
        + 0.36 * local_score
        + 0.18 * rank_score
        + 0.07 * regret_score
        + 0.05 * mutual_nearest.astype(float)
    )

    score = np.where(candidate, score, 0.84 * score)
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score[~np.isfinite(score)] = 0.0
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
