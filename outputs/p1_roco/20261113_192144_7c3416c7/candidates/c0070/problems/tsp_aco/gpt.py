import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inv_distance = 1.0 / (work + 1e-10 * scale)
    inv_distance[~np.isfinite(inv_distance)] = 0.0

    row_max = np.max(inv_distance, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    normalized_inverse = inv_distance / row_max

    order = np.argsort(work, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    finite_mask = np.isfinite(work)
    finite_counts = np.sum(finite_mask, axis=1)
    neighborhood_mean = (
        np.sum(np.where(finite_mask, work, 0.0), axis=1)
        / np.maximum(finite_counts, 1)
    )

    local_scale = np.sqrt(
        neighborhood_mean[:, None] * neighborhood_mean[None, :]
    )
    local_scale = np.maximum(local_scale, scale * 1e-10)

    contrast = np.exp(-work / local_scale)
    contrast[~np.isfinite(contrast)] = 0.0

    score = (
        0.50 * normalized_inverse
        + 0.32 * mutual_rank
        + 0.18 * contrast
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.power(np.maximum(score, 0.0), 1.08)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
