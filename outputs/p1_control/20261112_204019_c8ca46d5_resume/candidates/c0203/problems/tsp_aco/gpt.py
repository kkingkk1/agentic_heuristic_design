import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    positive = d[finite_positive]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(np.isfinite(d), d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(1.25 * np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    scale = np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps

    relative_length = d / scale
    closeness = np.exp(-relative_length)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    candidate_count = max(1, int(np.ceil(0.25 * (n - 1))))
    candidate_order = np.argpartition(work, candidate_count - 1, axis=1)
    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[row_index, candidate_order[:, :candidate_count]] = True
    np.fill_diagonal(candidate_mask, False)

    mutual_candidate = candidate_mask & candidate_mask.T

    if n >= 3:
        first_two = np.partition(work, 1, axis=1)[:, :2]
        alternative = np.maximum(first_two[:, 1], eps)
    else:
        alternative = np.maximum(work[:, 0], eps)

    endpoint_regret = np.maximum(
        0.0,
        (alternative[:, None] - d) / (local_scale[:, None] + eps),
    )
    endpoint_regret += np.maximum(
        0.0,
        (alternative[None, :] - d) / (local_scale[None, :] + eps),
    )
    endpoint_regret = np.minimum(endpoint_regret, 2.0)

    indispensability = 1.0 - np.exp(-endpoint_regret)

    score = closeness * mutual_rank
    score *= 1.0 + 0.12 * indispensability
    score *= 1.0 + 0.08 * mutual_candidate.astype(float)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
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
