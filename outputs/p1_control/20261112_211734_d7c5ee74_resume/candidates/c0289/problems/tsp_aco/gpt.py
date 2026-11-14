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

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    local = np.partition(work, k - 1, axis=1)[:, :k]
    kth = np.maximum(local[:, -1], eps)
    local_mean = np.maximum(local.mean(axis=1), eps)

    pair_scale = np.sqrt(local_mean[:, None] * local_mean[None, :])
    relative_length = d / (pair_scale + eps)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    length_score = np.exp(-1.12 * np.minimum(relative_length, 40.0))

    alternative = np.maximum(kth, eps)
    endpoint_regret = alternative[:, None] / (d + alternative[:, None] + eps)
    reverse_regret = alternative[None, :] / (d + alternative[None, :] + eps)
    regret = np.sqrt(endpoint_regret * reverse_regret)

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    density = np.sqrt(
        local_mean[:, None] * local_mean[None, :]
    ) / (np.sqrt(kth[:, None] * kth[None, :]) + eps)
    density_factor = 0.94 + 0.06 * np.clip(density, 0.0, 1.0)

    score = (
        mutual_rank
        * length_score
        * (0.82 + 0.18 * mutual_competitiveness)
        * (0.88 + 0.12 * regret)
        * density_factor
    )

    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 84.0))
        score *= np.where(score >= cutoff, 1.028, 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
