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
    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    order = np.argsort(work, axis=1, kind="mergesort")
    nearest_idx = order[:, :k]
    nearest = np.take_along_axis(work, nearest_idx, axis=1)

    local_scale = np.maximum(nearest.mean(axis=1), eps)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    kth = nearest[:, -1]
    next_kth = (
        np.take_along_axis(work, order[:, : min(k + 1, n - 1)], axis=1)[:, -1]
        if n > 2
        else kth
    )

    relative_length = d / (pair_scale + eps)
    length_score = np.exp(-1.15 * relative_length)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        np.maximum(kth[:, None], eps) * np.maximum(kth[None, :], eps)
    )
    margin = substitute_scale / (d + substitute_scale + eps)

    endpoint_pressure = np.sqrt(
        (kth[:, None] / (d + kth[:, None] + eps))
        * (kth[None, :] / (d + kth[None, :] + eps))
    )

    neighbor_mask = np.zeros((n, n), dtype=float)
    rows = np.repeat(np.arange(n), k)
    cols = nearest_idx.ravel()
    neighbor_mask[rows, cols] = 1.0
    shared_neighbor = (neighbor_mask @ neighbor_mask.T) / float(k)
    shared_neighbor = np.clip(shared_neighbor, 0.0, 1.0)

    triangle_bonus = 0.94 + 0.06 * shared_neighbor

    score = (
        length_score
        * mutual_rank**1.55
        * (0.80 + 0.20 * mutual_competitiveness)
        * (0.90 + 0.10 * margin)
        * (0.92 + 0.08 * endpoint_pressure)
        * triangle_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 84.0))
        score *= np.where(score >= cutoff, 1.03, 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
