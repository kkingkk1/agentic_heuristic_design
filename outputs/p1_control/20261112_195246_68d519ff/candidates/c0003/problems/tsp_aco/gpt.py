import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return prior indicators for promising TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros_like(distances, dtype=float)

    d = distances.copy()
    d[~np.isfinite(d)] = np.inf
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    closeness = np.exp(-d / scale)
    closeness[~np.isfinite(closeness)] = 0.0
    np.fill_diagonal(closeness, 0.0)

    row_sums = closeness.sum(axis=1, keepdims=True)
    probabilities = closeness / np.maximum(row_sums, np.finfo(float).eps)

    mutuality = np.sqrt(probabilities * probabilities.T)

    if n > 2:
        local_density = probabilities.sum(axis=1)
        isolation = 1.0 / np.maximum(local_density, np.finfo(float).eps)
        isolation /= np.maximum(isolation.max(), np.finfo(float).eps)
        endpoint_factor = np.sqrt(isolation[:, None] * isolation[None, :])
    else:
        endpoint_factor = np.ones((n, n), dtype=float)

    score = closeness * (0.5 + 0.5 * mutuality) * (0.75 + 0.25 * endpoint_factor)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = score.max(axis=1, keepdims=True)
    score /= np.maximum(row_max, np.finfo(float).eps)

    return score
