import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid_directed = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid_directed, False)

    valid = valid_directed & valid_directed.T
    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    order = np.argsort(pair_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    nearest = np.min(pair_d, axis=1)
    nearest = np.where(np.isfinite(nearest) & (nearest > 0), nearest, np.nan)

    positive_edges = pair_d[valid & (pair_d > 0)]
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size:
        global_scale = float(np.median(finite_nearest))
    elif positive_edges.size:
        global_scale = float(np.median(positive_edges))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)
    local_scale = np.where(np.isfinite(nearest), nearest, global_scale)

    safe_pair_d = np.where(valid, pair_d, np.inf)

    global_ratio = safe_pair_d / global_scale
    global_score = np.exp(-np.minimum(global_ratio, 40.0))

    local_ratio = safe_pair_d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_score = np.exp(-np.minimum(local_ratio, 40.0))

    rank_score = np.exp(-0.68 * np.minimum(ranks, n))
    rank_score = np.where(valid, rank_score, 0.0)

    candidate_limit = min(4, max(1, n - 1))
    candidate = valid & (ranks <= candidate_limit)

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    second = np.full(n, np.nan, dtype=float)
    if n >= 3:
        second = safe_pair_d[
            np.arange(n),
            order[:, 1],
        ]

    has_two = np.isfinite(second) & np.isfinite(nearest) & (second > 0)
    regret = np.zeros(n, dtype=float)
    regret[has_two] = np.clip(
        (second[has_two] - nearest[has_two])
        / np.maximum(second[has_two], np.finfo(float).eps),
        0.0,
        1.0,
    )

    nearest_regret = (
        nearest_edge
        * 0.5
        * (regret[:, None] + regret[None, :])
    )

    row_column_competition = np.zeros_like(pair_d, dtype=float)
    top_two = valid & (ranks <= 1)
    row_column_competition[top_two] = np.exp(
        -0.5 * (ranks[top_two] + ranks.T[top_two])
    )

    score = (
        0.28 * global_score
        + 0.34 * local_score
        + 0.20 * rank_score
        + 0.12 * nearest_regret
        + 0.16 * mutual_nearest.astype(float)
        + 0.06 * row_column_competition
    )

    score = np.where(candidate, score, 0.84 * score)
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
