import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-prior indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    inv_distance = 1.0 / (work + eps)
    inv_distance[~np.isfinite(inv_distance)] = 0.0

    order = np.argsort(work, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)

    nearest = order[:, 0]
    mutual = np.equal(nearest[:, None], np.arange(n)[None, :])
    mutual = mutual | mutual.T
    mutual_score = mutual.astype(float)

    row_mean = np.mean(np.where(np.isfinite(work), work, 0.0), axis=1)
    row_mean = np.maximum(row_mean, eps)
    savings = (row_mean[:, None] + row_mean[None, :] - work)
    savings = np.maximum(savings, 0.0)
    savings /= np.maximum(row_mean[:, None] + row_mean[None, :], eps)

    score = (
        0.55 * inv_distance / max(np.max(inv_distance), eps)
        + 0.20 * rank_score / max(np.max(rank_score), eps)
        + 0.15 * savings
        + 0.10 * mutual_score
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
