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

    # Robust endpoint-local distance normalization.
    relative_i = d / local_scale[:, None]
    relative_j = d / local_scale[None, :]
    relative = np.sqrt(
        np.maximum(relative_i, eps) * np.maximum(relative_j, eps)
    )

    distance_attraction = np.exp(-np.minimum(relative, 12.0))

    # Stable reciprocal ranks with deterministic tie handling.
    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(m + 1, dtype=float)

    rank_i = 1.0 / (1.0 + ranks)
    reciprocal_rank = np.sqrt(rank_i * rank_i.T)

    # Soft candidate membership based on both endpoint ranks.
    rank_scale = max(float(k), 1.0)
    rank_pressure = 0.5 * (
        ranks / rank_scale + ranks.T / rank_scale
    )
    candidate_gate = 0.84 + 0.16 * np.exp(
        -np.minimum(rank_pressure, 8.0)
    )

    # Bounded endpoint competition.
    competition_i = 1.0 / (1.0 + relative_i)
    competition_j = 1.0 / (1.0 + relative_j)
    competition = np.sqrt(competition_i * competition_j)
    competition_factor = 0.84 + 0.16 * competition

    # Reward edges that clearly beat the local alternative at both endpoints.
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
    margin_factor = 0.95 + 0.05 * margin

    # Common-neighborhood support: both endpoints should regard nearby
    # nodes similarly, but the effect remains deliberately weak.
    local_affinity = np.exp(
        -np.minimum(work / local_scale[:, None], 12.0)
    )
    np.fill_diagonal(local_affinity, 0.0)

    support = local_affinity @ local_affinity.T
    support_scale = np.sqrt(
        np.maximum(
            np.sum(local_affinity * local_affinity, axis=1)[:, None]
            * np.sum(local_affinity * local_affinity, axis=1)[None, :],
            eps,
        )
    )
    normalized_support = np.clip(support / (support_scale + eps), 0.0, 1.0)
    support_factor = 0.97 + 0.03 * normalized_support

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
