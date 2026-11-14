import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    np.fill_diagonal(valid, False)

    valid = valid & valid.T
    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)

    order = np.argsort(pair_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    positive = pair_d[valid & (pair_d > 0.0)]
    if positive.size:
        distance_scale = float(np.median(positive))
    else:
        distance_scale = 1.0
    distance_scale = max(distance_scale, np.finfo(float).eps)

    nearest = pair_d[np.arange(n), order[:, 0]]
    nearest = np.where(np.isfinite(nearest), nearest, distance_scale)

    local_scale = np.maximum(nearest, distance_scale * 1e-6)
    normalized_distance = pair_d / distance_scale
    local_distance = pair_d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    distance_score = np.exp(-np.minimum(normalized_distance, 40.0))
    local_score = np.exp(-np.minimum(local_distance, 40.0))

    rank_score = np.exp(
        -0.72 * np.minimum(ranks.astype(float), float(n))
    )
    rank_score = np.where(valid, rank_score, 0.0)

    candidate_width = min(5, max(2, n // 8), n - 1)
    candidate = valid & (ranks < candidate_width)

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    if n >= 3:
        second = pair_d[np.arange(n), order[:, 1]]
        has_two = np.isfinite(nearest) & np.isfinite(second)
        regret = np.zeros(n, dtype=float)
        regret[has_two] = np.clip(
            (second[has_two] - nearest[has_two])
            / np.maximum(second[has_two], np.finfo(float).eps),
            0.0,
            1.0,
        )
    else:
        regret = np.zeros(n, dtype=float)

    regret_score = nearest_edge * (
        0.5 * (regret[:, None] + regret[None, :])
    )

    mutual_rank_score = np.exp(
        -0.35 * (ranks.astype(float) + ranks.T.astype(float))
    )
    mutual_rank_score = np.where(
        valid & (ranks < candidate_width) & (ranks.T < candidate_width),
        mutual_rank_score,
        0.0,
    )

    score = (
        0.30 * distance_score
        + 0.30 * local_score
        + 0.20 * rank_score
        + 0.10 * regret_score
        + 0.07 * mutual_nearest.astype(float)
        + 0.03 * mutual_rank_score
    )

    score = np.where(candidate, score, 0.82 * score)
    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
