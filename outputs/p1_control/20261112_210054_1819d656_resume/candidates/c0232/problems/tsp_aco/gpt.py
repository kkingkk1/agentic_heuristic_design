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

    neighborhood = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(neighborhood.mean(axis=1), eps)
    substitute = np.maximum(neighborhood[:, -1], eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = d / (pair_scale + eps)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks = np.minimum(ranks, float(n - 1))

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    distance_signal = np.exp(-0.9 * np.minimum(relative_length, 50.0))

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        substitute[:, None] * substitute[None, :]
    )
    substitute_margin = substitute_scale / (
        d + substitute_scale + eps
    )

    local_regret = np.minimum(
        np.maximum(substitute[:, None] - d, 0.0)
        / (substitute[:, None] + eps),
        1.0,
    )
    local_regret = np.minimum(
        local_regret
        + np.maximum(substitute[None, :] - d, 0.0)
        / (substitute[None, :] + eps),
        1.0,
    )

    score = (
        mutual_rank ** 1.75
        * distance_signal
        * (0.78 + 0.22 * mutual_competitiveness)
        * (0.965 + 0.035 * substitute_margin)
        * (1.0 + 0.035 * local_regret)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    mask = ~np.eye(n, dtype=bool)
    values = score[mask]

    if values.size:
        cutoff = float(np.percentile(values, 82.0))
        score *= np.where(score >= cutoff, 1.035, 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
