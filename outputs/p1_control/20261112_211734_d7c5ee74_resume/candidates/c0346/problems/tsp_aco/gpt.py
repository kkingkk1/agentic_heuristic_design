import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    fallback = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )

    d = np.where(np.isfinite(d), d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    m = n - 1
    k = max(1, min(m, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        eps,
    )
    kth_distance = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    relative_distance = d / pair_scale
    attraction = np.exp(-np.minimum(relative_distance, 12.0))

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    mutual_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    mutual_rank /= np.maximum(np.max(mutual_rank), eps)

    endpoint_ratio_i = d / np.maximum(local_scale[:, None], eps)
    endpoint_ratio_j = d / np.maximum(local_scale[None, :], eps)

    competition = 1.0 / np.sqrt(
        (1.0 + endpoint_ratio_i)
        * (1.0 + endpoint_ratio_j)
    )

    rank_pressure = 0.5 * (
        ranks / max(float(k), 1.0)
        + ranks.T / max(float(k), 1.0)
    )
    candidate_prior = np.exp(
        -0.5 * np.minimum(rank_pressure, 12.0)
    )

    regret_i = np.clip(
        (kth_distance[:, None] - d)
        / (kth_distance[:, None] + d + eps),
        0.0,
        1.0,
    )
    regret_j = np.clip(
        (kth_distance[None, :] - d)
        / (kth_distance[None, :] + d + eps),
        0.0,
        1.0,
    )
    regret = 0.5 * (regret_i + regret_j)

    affinity = np.exp(
        -np.minimum(work / np.maximum(local_scale[:, None], eps), 12.0)
    )
    np.fill_diagonal(affinity, 0.0)

    row_norm = np.sqrt(
        np.maximum(np.sum(affinity * affinity, axis=1), eps)
    )
    compatibility = (
        affinity @ affinity.T
    ) / np.maximum(
        row_norm[:, None] * row_norm[None, :],
        eps,
    )
    compatibility = np.clip(compatibility, 0.0, 1.0)

    score = (
        1.00 * np.log(np.maximum(attraction, eps))
        + 0.85 * np.log(np.maximum(mutual_rank, eps))
        + 0.55 * np.log(np.maximum(candidate_prior, eps))
        + 0.45 * np.log(np.maximum(competition, eps))
        + 0.35 * regret
        + 0.30 * compatibility
    )

    score -= np.max(score)
    score = np.exp(np.clip(score, -40.0, 0.0))
    score = 0.5 * (score + score.T)

    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
