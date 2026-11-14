import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid_directed = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid_directed, False)

    valid = valid_directed & valid_directed.T
    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    order = np.argsort(pair_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    positive = pair_d[valid & (pair_d > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
        dispersion = float(
            np.median(np.abs(positive - global_scale))
            / max(global_scale, np.finfo(float).eps)
        )
    else:
        global_scale = 1.0
        dispersion = 0.0

    global_scale = max(global_scale, np.finfo(float).eps)
    nearest = np.min(pair_d, axis=1)
    local_scale = np.where(
        np.isfinite(nearest),
        np.maximum(nearest, 0.25 * global_scale),
        global_scale,
    )

    safe_d = np.where(valid, pair_d, np.inf)
    temperature = float(np.clip(0.72 + 0.30 * dispersion, 0.65, 1.10))

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        global_affinity = np.exp(
            -np.minimum(safe_d / global_scale / temperature, 40.0)
        )
        local_ratio = safe_d / np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        )
        local_affinity = np.exp(
            -np.minimum(local_ratio / temperature, 40.0)
        )

    global_affinity = np.where(valid, global_affinity, 0.0)
    local_affinity = np.where(valid, local_affinity, 0.0)

    rank_affinity = np.exp(-np.minimum(ranks, n) / 1.35)
    rank_affinity = np.where(valid, rank_affinity, 0.0)

    candidate_limit = min(4, max(1, n - 1))
    candidate = valid & (ranks < candidate_limit)

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    regret = np.zeros(n, dtype=float)
    if n >= 3:
        second = safe_d[np.arange(n), order[:, 1]]
        has_two = (
            np.isfinite(nearest)
            & np.isfinite(second)
            & (second > 0.0)
        )
        regret[has_two] = np.clip(
            (second[has_two] - nearest[has_two])
            / np.maximum(second[has_two], np.finfo(float).eps),
            0.0,
            1.0,
        )

    regret_bonus = nearest_edge * 0.5 * (
        regret[:, None] + regret[None, :]
    )

    top_two = valid & (ranks <= 1)
    competition = np.zeros_like(pair_d, dtype=float)
    competition[top_two] = np.exp(
        -0.45 * (ranks[top_two] + ranks.T[top_two])
    )

    score = (
        0.22 * global_affinity
        + 0.28 * local_affinity
        + 0.22 * rank_affinity
        + 0.08 * regret_bonus
        + 0.14 * mutual_nearest.astype(float)
        + 0.06 * competition
    )

    floor_weight = float(
        np.clip(0.05 + 0.08 * dispersion / (1.0 + dispersion), 0.05, 0.13)
    )
    diversity_floor = 0.35 + 0.65 * rank_affinity
    score = (1.0 - floor_weight) * score + floor_weight * diversity_floor

    score = np.where(candidate, score, 0.88 * score)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
