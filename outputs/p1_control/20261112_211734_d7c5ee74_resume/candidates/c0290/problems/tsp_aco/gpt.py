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
    fallback = float(np.median(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    m = n - 1
    k = max(1, min(m, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    kth = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    endpoint_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    relative_distance = d / endpoint_scale

    attraction = np.exp(-np.minimum(relative_distance, 10.0))

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    rank_pressure = 0.5 * (
        ranks / max(float(k), 1.0)
        + ranks.T / max(float(k), 1.0)
    )
    candidate_prior = 0.82 + 0.18 * np.exp(
        -np.minimum(rank_pressure, 8.0)
    )

    relative_i = d / local_scale[:, None]
    relative_j = d / local_scale[None, :]

    competition = np.sqrt(
        1.0 / (1.0 + relative_i)
        * 1.0 / (1.0 + relative_j)
    )
    competition_factor = 0.86 + 0.14 * competition

    margin_i = (kth[:, None] - d) / (
        kth[:, None] + d + eps
    )
    margin_j = (kth[None, :] - d) / (
        kth[None, :] + d + eps
    )
    regret_margin = 0.5 * (
        np.clip(margin_i, 0.0, 1.0)
        + np.clip(margin_j, 0.0, 1.0)
    )
    margin_factor = 0.94 + 0.06 * regret_margin

    affinity = np.exp(
        -np.minimum(work / local_scale[:, None], 10.0)
    )
    np.fill_diagonal(affinity, 0.0)

    support = affinity @ affinity.T
    norm = np.sqrt(
        np.maximum(
            np.sum(affinity * affinity, axis=1)[:, None]
            * np.sum(affinity * affinity, axis=1)[None, :],
            eps,
        )
    )
    compatibility = np.clip(support / (norm + eps), 0.0, 1.0)
    compatibility_factor = 0.97 + 0.03 * compatibility

    score = (
        attraction
        * reciprocal_rank
        * candidate_prior
        * competition_factor
        * margin_factor
        * compatibility_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
