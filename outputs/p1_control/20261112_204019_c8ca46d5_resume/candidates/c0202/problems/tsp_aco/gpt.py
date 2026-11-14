import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(1.25 * np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps

    relative_length = d / pair_scale
    closeness = np.exp(-np.minimum(relative_length, 50.0))
    np.fill_diagonal(closeness, 0.0)

    row_mass = np.sum(closeness, axis=1, keepdims=True) + eps
    directional_competition = closeness / row_mass
    mutual_competition = np.sqrt(
        directional_competition * directional_competition.T
    )

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]

    reciprocal_rank = np.sqrt(
        (1.0 / (1.0 + ranks))
        * (1.0 / (1.0 + ranks.T))
    )

    competition_scale = np.max(mutual_competition)
    if competition_scale > eps:
        competition_term = mutual_competition / competition_scale
    else:
        competition_term = np.zeros_like(mutual_competition)

    score = (
        closeness
        * reciprocal_rank
        * (0.75 + 0.25 * competition_term)
    )

    detours = (
        work[:, None, :]
        + work[None, :, :]
        - d[:, :, None]
    )

    node_index = np.arange(n)
    valid_intermediate = (
        (node_index[None, None, :] != node_index[:, None, None])
        & (node_index[None, None, :] != node_index[None, :, None])
    )
    detours = np.where(valid_intermediate, detours, np.inf)

    alternative = np.min(detours, axis=2)
    margin = np.maximum(alternative - d, 0.0)
    margin = np.where(np.isfinite(margin), margin, 0.0)

    normalized_margin = margin / pair_scale
    clipped_margin = np.minimum(normalized_margin, 2.0) / 2.0

    candidate_count = max(1, int(np.ceil(0.20 * (n - 1))))
    short_candidate = ranks < candidate_count
    off_diagonal = ~np.eye(n, dtype=bool)
    mutual_short = short_candidate & short_candidate.T & off_diagonal

    margin_bonus = 0.06 * clipped_margin * mutual_short
    score *= 1.0 + margin_bonus

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
