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

    neighborhood = max(
        1,
        min(n - 1, int(np.ceil(1.5 * np.sqrt(n)))),
    )

    nearest = np.partition(
        work,
        neighborhood - 1,
        axis=1,
    )[:, :neighborhood]

    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) + eps

    relative_distance = d / pair_scale
    distance_score = np.exp(-relative_distance)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)

    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]

    rank_fraction = ranks / max(1, n - 1)
    rank_score = 1.0 / (1.0 + ranks)

    mutual_rank = np.sqrt(rank_score * rank_score.T)

    off_diagonal = ~np.eye(n, dtype=bool)

    candidate_count = max(
        1,
        min(n - 1, int(np.ceil(0.20 * (n - 1)))),
    )

    short_candidate = (
        (ranks < candidate_count)
        & off_diagonal
    )

    reciprocal_candidate = short_candidate & short_candidate.T

    rank_agreement = np.exp(
        -1.5 * np.abs(rank_fraction - rank_fraction.T)
    )

    reciprocal_strength = np.sqrt(
        np.clip(1.0 - rank_fraction, 0.0, 1.0)
        * np.clip(1.0 - rank_fraction.T, 0.0, 1.0)
    )

    smooth_bonus = (
        0.08
        * reciprocal_strength
        * rank_agreement
        * reciprocal_candidate
    )

    score = distance_score * mutual_rank
    score *= 1.0 + smooth_bonus

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    low = float(np.min(values))
    high = float(np.max(values))

    if high > low:
        score = (score - low) / (high - low)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
