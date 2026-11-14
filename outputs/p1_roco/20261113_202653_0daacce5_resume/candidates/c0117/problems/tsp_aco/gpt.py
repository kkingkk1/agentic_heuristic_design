import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be nonnegative")

    d = 0.5 * (d + d.T)
    valid = ~np.eye(n, dtype=bool)

    positive = d[valid & (d > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(global_scale, 1.0)

    positive_mask = valid & (d > 0)
    row_counts = positive_mask.sum(axis=1)

    row_median = np.zeros(n, dtype=float)
    for i in range(n):
        values = d[i, positive_mask[i]]
        row_median[i] = np.median(values) if values.size else global_scale

    row_scale = np.maximum(row_median, eps)

    safe_d = np.where(valid, d, np.inf)

    exponential = np.exp(-safe_d / row_scale[:, None])
    exponential[~valid] = 0.0
    exponential /= np.maximum(exponential.sum(axis=1, keepdims=True), eps)

    reciprocal = np.zeros_like(d)
    reciprocal[valid] = 1.0 / np.maximum(d[valid], eps)
    reciprocal /= np.maximum(reciprocal.sum(axis=1, keepdims=True), eps)

    local_best = np.min(np.where(valid, d, np.inf), axis=1)
    local_best = np.maximum(local_best, eps)

    contrast = np.zeros_like(d)
    contrast[valid] = np.exp(
        -np.maximum(d[valid] / local_best[:, None][valid] - 1.0, 0.0)
    )
    contrast /= np.maximum(contrast.sum(axis=1, keepdims=True), eps)

    mutual_exponential = np.sqrt(exponential * exponential.T)
    mutual_reciprocal = np.sqrt(reciprocal * reciprocal.T)
    mutual_contrast = np.sqrt(contrast * contrast.T)

    rank = np.argsort(np.where(valid, d, np.inf), axis=1, kind="stable")
    rank_bonus = np.zeros_like(d)

    ranks = np.arange(n, dtype=float)
    for i in range(n):
        ordered = rank[i]
        ordered = ordered[ordered != i]
        rank_bonus[i, ordered] = 1.0 / (1.0 + ranks[: ordered.size])

    rank_bonus = np.sqrt(rank_bonus * rank_bonus.T)

    score = (
        0.40 * mutual_exponential
        + 0.30 * mutual_reciprocal
        + 0.20 * mutual_contrast
        + 0.10 * rank_bonus
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    return score
