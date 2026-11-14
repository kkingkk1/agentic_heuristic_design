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
    k = max(1, min(n - 1, int(np.ceil(n ** 0.6))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(nearest.mean(axis=1), eps)
    alternative = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    endpoint_relative = d / (local_scale[:, None] + eps)
    symmetric_relative = 0.5 * (
        endpoint_relative + endpoint_relative.T
    )

    distance_attraction = np.exp(-0.92 * symmetric_relative)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + 0.92 * ranks)
    reciprocal_rank = np.sqrt(rank_score * rank_score.T)

    endpoint_competition = 1.0 / (1.0 + 0.92 * endpoint_relative)
    mutual_competition = np.sqrt(
        endpoint_competition * endpoint_competition.T
    )

    margin_i = (alternative[:, None] - d) / (
        alternative[:, None] + d + eps
    )
    margin_j = (alternative[None, :] - d) / (
        alternative[None, :] + d + eps
    )
    clipped_margin = 0.5 * (
        np.clip(margin_i, 0.0, 1.0)
        + np.clip(margin_j, 0.0, 1.0)
    )
    margin_factor = 0.94 + 0.06 * clipped_margin

    normalized_rank = np.minimum(ranks / float(k), 4.0)
    candidate_gate = 0.86 + 0.14 * np.exp(
        -0.5 * (normalized_rank + normalized_rank.T)
    )

    score = (
        distance_attraction
        * reciprocal_rank
        * (0.80 + 0.20 * mutual_competition)
        * margin_factor
        * candidate_gate
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
