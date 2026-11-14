import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-inclusion priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_off_diagonal = np.isfinite(distances)
    np.fill_diagonal(finite_off_diagonal, False)
    if not np.any(finite_off_diagonal):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[finite_off_diagonal]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0

    d = distances.copy()
    d[~np.isfinite(d)] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive_values = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse_distance = 1.0 / (1.0 + d / scale)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    nearest_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest_indices = np.argpartition(
        d, kth=nearest_count - 1, axis=1
    )[:, :nearest_count]

    nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    nearest[rows, nearest_indices] = 1.0
    np.fill_diagonal(nearest, 0.0)

    mutual = nearest * nearest.T
    one_sided = nearest + nearest.T - mutual

    score = (
        (0.55 / 0.75) * inverse_distance
        + (0.15 / 0.75) * mutual
        + (0.05 / 0.75) * one_sided
    )

    score = (score + score.T) / 2.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
