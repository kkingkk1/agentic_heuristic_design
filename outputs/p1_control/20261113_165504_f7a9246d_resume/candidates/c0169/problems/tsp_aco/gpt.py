import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    directed_valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    pair_d = np.full((n, n), np.inf, dtype=float)
    pair_d[valid] = 0.5 * (d[valid] + d.T[valid])

    order = np.argsort(pair_d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(pair_d, order, axis=1)

    ranks = np.full((n, n), n, dtype=int)
    row = np.arange(n)[:, None]
    ranks[row, order] = np.arange(n)[None, :]
    ranks[~valid] = n

    positive = pair_d[valid & (pair_d > 0.0)]
    if positive.size:
        base_scale = float(np.median(positive))
    else:
        base_scale = 1.0

    eps = np.finfo(float).eps
    base_scale = max(base_scale, eps)
    floor_scale = max(base_scale * 0.05, eps)

    nearest = sorted_d[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, base_scale)
    endpoint_scale = np.maximum(nearest, floor_scale)

    global_ratio = pair_d / base_scale
    global_score = np.exp(-np.minimum(global_ratio, 40.0))
    global_score[~valid] = 0.0

    local_denominator = np.sqrt(
        endpoint_scale[:, None] * endpoint_scale[None, :]
    )
    local_ratio = pair_d / np.maximum(local_denominator, eps)
    local_score = np.exp(-np.minimum(local_ratio, 40.0))
    local_score[~valid] = 0.0

    rank_score = np.exp(-0.62 * np.minimum(ranks, n))
    rank_score[~valid] = 0.0

    candidate_limit = min(6, max(1, int(np.ceil(np.sqrt(n)))))
    candidate = valid & (ranks < candidate_limit)

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    second = np.full(n, np.inf, dtype=float)
    if n >= 3:
        second = sorted_d[:, 1]

    has_two = np.isfinite(nearest) & np.isfinite(second) & (second > 0.0)
    regret = np.zeros(n, dtype=float)
    regret[has_two] = np.clip(
        (second[has_two] - nearest[has_two])
        / np.maximum(second[has_two], eps),
        0.0,
        1.0,
    )
    regret *= (regret >= 0.15)

    regret_bonus = 0.5 * (
        regret[:, None] + regret[None, :]
    )
    regret_bonus *= nearest_edge
    regret_bonus[~valid] = 0.0

    top_two = valid & (ranks <= 1)
    competition = np.zeros((n, n), dtype=float)
    competition[top_two] = np.exp(
        -0.45 * (ranks[top_two] + ranks.T[top_two])
    )

    score = (
        0.25 * global_score
        + 0.32 * local_score
        + 0.20 * rank_score
        + 0.10 * regret_bonus
        + 0.16 * mutual_nearest.astype(float)
        + 0.07 * competition
    )

    score = np.where(candidate, score, 0.78 * score)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
