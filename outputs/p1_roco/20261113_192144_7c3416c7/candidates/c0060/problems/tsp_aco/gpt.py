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
    global_scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(global_scale, 1.0)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    neighbor_count = min(max(2, int(np.sqrt(n))), n - 1)
    ordered = np.sort(d, axis=1)
    local_scale = ordered[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(np.outer(local_scale, local_scale))
    relative_distance = d / np.maximum(pair_scale, eps)

    adaptive_proximity = 1.0 / (1.0 + relative_distance)

    inverse_distance = 1.0 / np.maximum(d, eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    row_sum = inverse_distance.sum(axis=1, keepdims=True)
    inverse_distance /= np.maximum(row_sum, eps)

    density = global_scale / local_scale
    density /= max(float(density.max()), 1.0)
    density_bonus = np.sqrt(np.outer(density, density))

    score = (
        0.55 * adaptive_proximity
        + 0.45 * inverse_distance
    ) * (0.9 + 0.1 * density_bonus)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    return score
