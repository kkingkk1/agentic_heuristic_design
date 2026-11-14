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

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), np.finfo(float).eps)

    scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = d / (scale + np.finfo(float).eps)
    closeness = np.exp(-relative_length)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    log_scale = np.log(local_scale)
    bridge = np.abs(log_scale[:, None] - log_scale[None, :])
    bridge = 1.0 - np.exp(-bridge)

    score = closeness * (0.55 + 0.45 * mutual_rank) * (1.0 + 0.15 * bridge)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    mask = ~np.eye(n, dtype=bool)
    values = score[mask]
    lo = np.min(values)
    hi = np.max(values)

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
