import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0.0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))

    neighborhood = np.partition(work, k - 1, axis=1)[:, :k]
    local_mean = np.maximum(neighborhood.mean(axis=1), eps)
    local_median = np.maximum(
        np.partition(neighborhood, k // 2, axis=1)[:, k // 2],
        eps,
    )
    kth = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    local_scale = np.sqrt(local_mean * local_median)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = np.minimum(d / (pair_scale + eps), 40.0)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    length_score = np.exp(-1.25 * relative_length)

    endpoint_opportunity = np.minimum(
        kth[:, None] / (d + kth[:, None] + eps),
        kth[None, :] / (d + kth[None, :] + eps),
    )
    opportunity_score = 0.82 + 0.18 * endpoint_opportunity

    local_density = np.sqrt(
        (local_mean[:, None] * local_mean[None, :])
        / (kth[:, None] * kth[None, :] + eps)
    )
    local_density = np.clip(local_density, 0.0, 1.0)

    bridge_score = 0.94 + 0.06 * local_density

    rank_agreement = np.exp(
        -0.5 * np.abs(ranks - ranks.T)
        / (1.0 + ranks + ranks.T)
    )
    agreement_score = 0.90 + 0.10 * rank_agreement

    score = (
        mutual_rank
        * length_score
        * opportunity_score
        * bridge_score
        * agreement_score
    )

    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 85.0))
        score *= np.where(score >= cutoff, 1.022, 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
