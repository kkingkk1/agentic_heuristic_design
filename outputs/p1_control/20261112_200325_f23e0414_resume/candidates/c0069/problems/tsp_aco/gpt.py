import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n) + 1.0))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    scale = np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps

    relative_length = d / scale
    closeness = np.exp(-1.15 * relative_length)

    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    detours = work[:, :, None] + work[None, :, :] - d[:, :, None]
    detours = np.where(np.isfinite(detours), detours, np.inf)

    for i in range(n):
        detours[i, i, :] = np.inf
        detours[i, :, i] = np.inf

    replacement_margin = np.min(detours, axis=2)
    replacement_margin = np.maximum(replacement_margin, 0.0)
    indispensable = 1.0 - np.exp(-replacement_margin / scale)

    score = (
        0.55 * closeness
        + 0.32 * mutual_rank
        + 0.13 * indispensable
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]
    lo = np.min(values)
    hi = np.max(values)

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
