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
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    alternative = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = d / (pair_scale + eps)

    closeness = np.exp(-relative)

    rank_scale = np.sqrt(alternative[:, None] * alternative[None, :])
    soft_rank = 1.0 / (1.0 + d / (rank_scale + eps))
    mutual_rank = np.sqrt(soft_rank * soft_rank.T)

    competitiveness = 1.0 / (1.0 + relative)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    replacement_margin = rank_scale / (d + rank_scale + eps)
    margin_factor = 0.92 + 0.08 * replacement_margin

    score = (
        closeness
        * mutual_rank
        * (0.82 + 0.18 * mutual_competitiveness)
        * margin_factor
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

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
