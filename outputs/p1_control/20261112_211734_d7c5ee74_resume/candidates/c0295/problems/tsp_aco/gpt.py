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

    neighborhood = np.partition(work, k - 1, axis=1)
    nearest = neighborhood[:, :k]
    alternative = np.maximum(neighborhood[:, k - 1], eps)

    local_scale = np.maximum(nearest.mean(axis=1), eps)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    relative_length = d / (pair_scale + eps)
    closeness = np.exp(-1.12 * relative_length)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (d + substitute_scale + eps)
    bounded_margin = 0.90 + 0.10 * margin

    score = (
        closeness
        * mutual_rank
        * (0.82 + 0.18 * mutual_competitiveness)
        * bounded_margin
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 85.0))
        score *= np.where(score >= cutoff, 1.03, 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
