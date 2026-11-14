import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    safe_d = np.where(valid, d, np.inf)

    finite_values = d[valid & np.isfinite(d)]
    positive_values = finite_values[finite_values > 0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = float(np.median(finite_values))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n)[None, :]

    rank_valid = np.where(valid, ranks, n)
    nearest_rank = rank_valid == 0
    second_rank = rank_valid == 1
    third_rank = rank_valid == 2

    nearest = np.min(safe_d, axis=1)

    second_nearest = np.full(n, np.inf, dtype=float)
    if n > 1:
        second_nearest = np.partition(safe_d, 1, axis=1)[:, 1]

    finite_nearest = nearest[np.isfinite(nearest)]
    fallback = float(np.median(finite_nearest)) if finite_nearest.size else scale

    nearest_safe = np.where(np.isfinite(nearest), nearest, fallback)
    second_safe = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        nearest_safe + scale,
    )

    regret_gap = np.maximum(second_safe - nearest_safe, 0.0)
    regret_scale = np.median(regret_gap[regret_gap > 0])
    if not np.isfinite(regret_scale) or regret_scale <= 0:
        regret_scale = scale

    regret = np.clip(regret_gap / (regret_gap + regret_scale), 0.0, 1.0)
    regret_bonus = nearest_rank * np.maximum(regret[:, None], regret[None, :])

    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        fallback,
    )

    local_ratio = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_score = np.exp(-np.minimum(local_ratio, 40.0))

    global_ratio = d / scale
    global_score = np.exp(-np.minimum(global_ratio, 40.0))

    rank_score = np.exp(-0.62 * np.minimum(rank_valid, n))
    rank_score = np.where(valid, rank_score, 0.0)

    reciprocal_rank = np.minimum(rank_valid, rank_valid.T)
    reciprocal_score = np.exp(-0.42 * reciprocal_rank)
    reciprocal_score = np.where(valid & valid.T, reciprocal_score, 0.0)

    mutual_bonus = (
        nearest_rank
        & nearest_rank.T
        & valid
        & valid.T
    ).astype(float)

    candidate_width = min(4, max(2, int(np.ceil(np.sqrt(n)))))
    candidate_support = (rank_valid < candidate_width).astype(float)
    candidate_support = np.minimum(candidate_support, candidate_support.T)
    candidate_support = np.where(valid, candidate_support, 0.0)

    score = (
        0.27 * global_score
        + 0.27 * local_score
        + 0.19 * rank_score
        + 0.10 * reciprocal_score
        + 0.10 * mutual_bonus
        + 0.05 * regret_bonus
        + 0.08 * candidate_support
    )

    outside_candidates = valid & (rank_valid >= candidate_width)
    score[outside_candidates] *= 0.88

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
