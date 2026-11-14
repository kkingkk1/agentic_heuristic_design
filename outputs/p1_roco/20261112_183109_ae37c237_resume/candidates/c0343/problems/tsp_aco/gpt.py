import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    values = d[valid]
    positive = values[values > 0.0]

    eps = np.finfo(float).eps
    if positive.size:
        global_scale = float(np.median(positive))
    elif values.size:
        global_scale = float(np.max(values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(values))
    penalty = max(finite_max, global_scale) * 8.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_work = np.take_along_axis(work, order, axis=1)
    best = sorted_work[:, 0]

    if n > 2:
        second = sorted_work[:, 1]
        third = sorted_work[:, 2]
    else:
        second = best
        third = best

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    relative_i = work / local_scale[:, None]
    relative_j = work / local_scale[None, :]
    relative_i = np.nan_to_num(
        relative_i,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    relative_j = np.nan_to_num(
        relative_j,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )

    geometric_relative = np.sqrt(
        np.maximum(relative_i, 0.0) * np.maximum(relative_j, 0.0)
    )

    distance_term = np.exp(
        -np.clip(0.92 * geometric_relative, 0.0, 60.0)
    )

    rank_scale = max(1.0, 0.68 * np.sqrt(float(n)))
    rank_i = np.exp(-ranks / rank_scale)
    rank_j = np.exp(-ranks.T / rank_scale)
    rank_term = np.sqrt(rank_i * rank_j)

    candidate_width = max(2, min(n - 1, int(np.ceil(np.sqrt(float(n))) + 1)))

    candidate_i = (
        ranks <= float(candidate_width)
    ) & valid
    candidate_j = candidate_i.T

    candidate_strength_i = np.exp(
        -np.clip(relative_i / 1.35, 0.0, 24.0)
    )
    candidate_strength_j = np.exp(
        -np.clip(relative_j / 1.35, 0.0, 24.0)
    )
    candidate_strength_i[~valid] = 0.0
    candidate_strength_j[~valid] = 0.0

    support_i = np.sum(
        candidate_strength_i * candidate_i,
        axis=1,
    )
    support_j = np.sum(
        candidate_strength_j * candidate_j,
        axis=1,
    )

    support_reference = np.median(
        np.concatenate(
            [
                support_i[support_i > 0.0],
                support_j[support_j > 0.0],
            ]
        )
    ) if np.any((support_i > 0.0) | (support_j > 0.0)) else 1.0

    if not np.isfinite(support_reference) or support_reference <= 0.0:
        support_reference = 1.0

    need_i = np.sqrt(
        support_reference / np.maximum(support_i, eps)
    )
    need_j = np.sqrt(
        support_reference / np.maximum(support_j, eps)
    )

    need_i = np.clip(need_i, 0.86, 1.22)
    need_j = np.clip(need_j, 0.86, 1.22)

    endpoint_need = np.sqrt(
        need_i[:, None] * need_j[None, :]
    )

    regret_i = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret_j = np.divide(
        third - second,
        np.maximum(third, eps),
    )

    regret_i = np.clip(np.nan_to_num(regret_i), 0.0, 1.0)
    regret_j = np.clip(np.nan_to_num(regret_j), 0.0, 1.0)

    regret_term = 1.0 + 0.24 * (
        regret_i[:, None] + regret_i[None, :]
    )
    regret_term *= 1.0 + 0.10 * (
        regret_j[:, None] + regret_j[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_term = np.where(mutual, 1.30, 1.0)

    shared_candidate_count = (
        candidate_i.astype(float) @ candidate_j.astype(float)
    )

    endpoint_degree = (
        np.sum(candidate_i, axis=1)
        + np.sum(candidate_j, axis=0)
    )

    coherence = np.divide(
        shared_candidate_count,
        np.maximum(0.5 * endpoint_degree, 1.0),
    )
    coherence = np.clip(coherence, 0.0, 1.0)

    bridge_term = 1.0 + 0.045 * coherence

    long_edge_penalty = np.exp(
        -0.58 * np.maximum(geometric_relative - 1.0, 0.0)
        -0.16 * np.maximum(relative_i + relative_j - 3.0, 0.0)
    )

    candidate_gate = np.where(
        candidate_i | candidate_j,
        1.0,
        0.34 * np.exp(-0.25 * np.maximum(geometric_relative - 1.0, 0.0)),
    )

    base_term = (
        0.52
        + 0.31 * rank_term
        + 0.17 * np.sqrt(
            np.exp(-np.clip(0.55 * relative_i, 0.0, 40.0))
            * np.exp(-np.clip(0.55 * relative_j, 0.0, 40.0))
        )
    )

    scores = (
        distance_term
        * base_term
        * endpoint_need
        * regret_term
        * mutual_term
        * bridge_term
        * long_edge_penalty
        * candidate_gate
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
