import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
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

    pair_d = np.full((n, n), np.inf, dtype=float)
    pair_d[valid] = 0.5 * (d[valid] + d.T[valid])

    order = np.argsort(pair_d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(pair_d, order, axis=1)

    degree = np.sum(valid, axis=1)
    max_candidates = min(6, max(1, n - 1))

    candidate_count = np.minimum(degree, max_candidates)
    candidate = np.zeros((n, n), dtype=bool)

    for i in range(n):
        count = int(candidate_count[i])
        if count > 0:
            candidate[i, order[i, :count]] = True

    candidate &= valid
    mutual_candidate = candidate & candidate.T

    positive = pair_d[valid & (pair_d > 0.0)]
    if positive.size:
        positive_floor = float(np.percentile(positive, 10.0))
        global_scale = float(np.median(positive))
    else:
        positive_floor = 1.0
        global_scale = 1.0

    positive_floor = max(positive_floor, np.finfo(float).eps)
    global_scale = max(global_scale, positive_floor)

    local_scale = np.empty(n, dtype=float)
    for i in range(n):
        row = sorted_d[i, : int(candidate_count[i])]
        row = row[np.isfinite(row)]

        positive_row = row[row > 0.0]
        if positive_row.size:
            local_scale[i] = max(float(np.median(positive_row)), positive_floor)
        elif row.size:
            local_scale[i] = positive_floor
        else:
            local_scale[i] = global_scale

    safe_d = np.where(valid, pair_d, np.inf)

    global_affinity = np.exp(
        -np.minimum(safe_d / global_scale, 40.0)
    )

    local_denominator = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_affinity = np.exp(
        -np.minimum(safe_d / local_denominator, 40.0)
    )

    rank = np.full((n, n), n, dtype=float)
    for i in range(n):
        valid_indices = order[i, np.isfinite(sorted_d[i])]
        rank[i, valid_indices] = np.arange(valid_indices.size)

    rank_affinity = np.exp(
        -0.72 * np.minimum(rank, float(n))
    )
    rank_affinity[~valid] = 0.0

    nearest = sorted_d[:, 0]
    nearest_valid = np.isfinite(nearest)

    second = np.full(n, np.nan, dtype=float)
    if n >= 3:
        second = sorted_d[:, 1]

    has_two = (
        nearest_valid
        & np.isfinite(second)
        & (second >= nearest)
        & (second > 0.0)
    )

    regret = np.zeros(n, dtype=float)
    regret[has_two] = np.clip(
        (second[has_two] - nearest[has_two])
        / np.maximum(second[has_two], positive_floor),
        0.0,
        1.0,
    )

    nearest_edge = valid & (rank == 0)
    selective_regret = (
        nearest_edge
        * 0.5
        * (regret[:, None] + regret[None, :])
    )

    mutual_nearest = nearest_edge & nearest_edge.T

    candidate_strength = (
        candidate.astype(float)
        + candidate.T.astype(float)
        + mutual_candidate.astype(float)
    )
    candidate_strength = np.clip(candidate_strength, 0.0, 3.0) / 3.0

    score = (
        0.24 * global_affinity
        + 0.34 * local_affinity
        + 0.22 * rank_affinity
        + 0.10 * selective_regret
        + 0.07 * mutual_nearest.astype(float)
        + 0.03 * candidate_strength
    )

    candidate_pair = candidate | candidate.T
    score = np.where(
        candidate_pair,
        score,
        0.68 * score + 0.32 * global_affinity,
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
