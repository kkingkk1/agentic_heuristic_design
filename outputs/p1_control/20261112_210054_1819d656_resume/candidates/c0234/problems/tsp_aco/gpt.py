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
    fallback = max(fallback, np.finfo(float).eps)

    d = np.where(np.isfinite(d), d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        eps,
    )
    nearest_edge = np.maximum(nearest[:, 0], eps)
    alternative = np.maximum(nearest[:, -1], eps)

    relative = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative = np.maximum(relative, 0.0)

    attraction = np.exp(-np.minimum(relative, 40.0))

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    rank_decay = 1.0 / np.sqrt(1.0 + ranks)
    reciprocal_rank = np.sqrt(rank_decay * rank_decay.T)

    rank_scale = max(float(k), 1.0)
    rank_gate = 0.90 + 0.10 * np.exp(
        -0.5
        * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
    )

    nearest_strength = np.exp(
        -np.minimum(
            d / np.maximum(nearest_edge[:, None], eps),
            40.0,
        )
    )
    mutual_nearest = np.sqrt(
        nearest_strength * nearest_strength.T
    )
    local_support = 0.86 + 0.14 * mutual_nearest

    margin_i = (
        alternative[:, None] - d
    ) / (
        alternative[:, None] + d + eps
    )
    margin_j = (
        alternative[None, :] - d
    ) / (
        alternative[None, :] + d + eps
    )

    margin = 0.5 * (
        np.clip(margin_i, 0.0, 1.0)
        + np.clip(margin_j, 0.0, 1.0)
    )
    regret_factor = 0.96 + 0.04 * margin

    score = (
        attraction
        * reciprocal_rank
        * local_support
        * regret_factor
        * rank_gate
    )

    score = np.maximum(score, 0.0)
    score = 0.5 * (score + score.T)

    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score = np.sqrt(np.clip(score, 0.0, 1.0))
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
