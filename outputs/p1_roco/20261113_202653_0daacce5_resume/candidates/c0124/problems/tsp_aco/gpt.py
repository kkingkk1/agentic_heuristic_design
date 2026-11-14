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
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    valid = np.isfinite(d)
    affinity = np.zeros_like(d, dtype=float)
    affinity[valid] = 1.0 / np.maximum(d[valid], eps)

    row_sum = affinity.sum(axis=1, keepdims=True)
    normalized_affinity = affinity / np.maximum(row_sum, eps)

    if n > 1:
        neighborhood_size = min(7, n - 1)
        nearest = np.partition(
            np.where(valid, d, np.inf),
            neighborhood_size - 1,
            axis=1,
        )[:, :neighborhood_size]

        mean_nearest = np.mean(nearest, axis=1)
        local_density = 1.0 / np.maximum(mean_nearest, eps)
        local_density /= max(float(local_density.max()), eps)
    else:
        local_density = np.zeros(1, dtype=float)

    density_bonus = np.sqrt(np.outer(local_density, local_density))

    score = normalized_affinity * (0.90 + 0.10 * density_bonus)
    score = 0.5 * (score + score.T)

    row_sum = score.sum(axis=1, keepdims=True)
    score /= np.maximum(row_sum, eps)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    return score
