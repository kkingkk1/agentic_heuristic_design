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

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = (d + d.T) * 0.5
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(finite_positive)) if finite_positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    valid = np.isfinite(d)
    reciprocal = np.zeros_like(d)
    reciprocal[valid] = 1.0 / np.maximum(d[valid], eps)

    row_totals = reciprocal.sum(axis=1, keepdims=True)
    affinity = reciprocal / np.maximum(row_totals, eps)

    neighborhood_size = min(5, n - 1)
    nearest = np.partition(
        np.where(valid, d, np.inf),
        neighborhood_size - 1,
        axis=1,
    )[:, :neighborhood_size]

    mean_nearest = nearest.mean(axis=1)
    density = 1.0 / np.maximum(mean_nearest, eps)
    density /= max(float(density.max()), eps)

    bonus = 0.85 + 0.15 * np.sqrt(density[:, None] * density[None, :])
    score = affinity * bonus
    score = (score + score.T) * 0.5

    totals = score.sum(axis=1, keepdims=True)
    score /= np.maximum(totals, eps)
    score = (score + score.T) * 0.5

    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    return score
