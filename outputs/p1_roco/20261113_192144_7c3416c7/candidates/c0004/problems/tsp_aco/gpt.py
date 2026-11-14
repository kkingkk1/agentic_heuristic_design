import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    if not np.all(finite):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Basic inverse-distance desirability.
    inv_distance = 1.0 / (work + 1e-12 * scale)
    inv_distance[~np.isfinite(inv_distance)] = 0.0

    # Normalize each row so values are comparable across nodes.
    row_max = np.max(inv_distance, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    normalized_inverse = inv_distance / row_max

    # Prefer edges that rank highly in both endpoint neighborhoods.
    order = np.argsort(work, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    # Emphasize edges that are short relative to both endpoint neighborhoods.
    neighborhood_mean = np.sum(
        np.where(np.isfinite(work), work, 0.0), axis=1
    ) / np.maximum(np.sum(np.isfinite(work), axis=1), 1)
    contrast = np.exp(
        -work / (np.sqrt(neighborhood_mean[:, None] * neighborhood_mean[None, :]) + scale * 1e-12)
    )
    contrast[~np.isfinite(contrast)] = 0.0

    score = (
        0.55 * normalized_inverse
        + 0.30 * mutual_rank
        + 0.15 * contrast
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
