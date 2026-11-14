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

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    m = n - 1
    k = max(1, min(m, int(np.ceil(1.15 * np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(
        0.9 * np.mean(nearest, axis=1)
        + 0.1 * np.median(nearest, axis=1),
        eps,
    )

    kth_distance = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    relative_i = d / local_scale[:, None]
    relative_j = d / local_scale[None, :]
    relative = np.sqrt(
        np.maximum(relative_i, eps) * np.maximum(relative_j, eps)
    )

    distance_attraction = np.exp(-np.minimum(relative, 10.0))

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(m + 1, dtype=float)

    rank_i = 1.0 / (1.0 + ranks)
    reciprocal_rank = np.sqrt(rank_i * rank_i.T)

    rank_scale = max(float(k), 1.0)
    rank_pressure = 0.5 * (
        ranks / rank_scale + ranks.T / rank_scale
    )
    candidate_gate = 0.80 + 0.20 * np.exp(
        -np.minimum(1.15 * rank_pressure, 8.0)
    )

    competition_i = 1.0 / (1.0 + 0.9 * relative_i)
    competition_j = 1.0 / (1.0 + 0.9 * relative_j)
    competition = np.sqrt(competition_i * competition_j)
    competition_factor = 0.82 + 0.18 * competition

    margin_i = (kth_distance[:, None] - d) / (
        kth_distance[:, None] + d + eps
    )
    margin_j = (kth_distance[None, :] - d) / (
        kth_distance[None, :] + d + eps
    )
    margin = 0.5 * (
        np.clip(margin_i, 0.0, 1.0)
        + np.clip(margin_j, 0.0, 1.0)
    )
    margin_factor = 0.94 + 0.06 * margin

    local_affinity = np.exp(
        -np.minimum(0.9 * work / local_scale[:, None], 12.0)
    )
    np.fill_diagonal(local_affinity, 0.0)

    support = local_affinity @ local_affinity.T
    support_norm = np.sqrt(
        np.maximum(
            np.sum(local_affinity * local_affinity, axis=1)[:, None]
            * np.sum(local_affinity * local_affinity, axis=1)[None, :],
            eps,
        )
    )
    normalized_support = np.clip(
        support / (support_norm + eps),
        0.0,
        1.0,
    )
    support_factor = 0.965 + 0.035 * normalized_support

    score = (
        distance_attraction
        * reciprocal_rank
        * competition_factor
        * margin_factor
        * candidate_gate
        * support_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score = score / maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
