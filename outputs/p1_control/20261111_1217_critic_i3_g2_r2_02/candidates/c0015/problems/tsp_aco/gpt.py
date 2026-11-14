import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-inclusion priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_mask = np.isfinite(distances)
    finite_off_diagonal = finite_mask.copy()
    np.fill_diagonal(finite_off_diagonal, False)

    if not np.any(finite_off_diagonal):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[finite_off_diagonal]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0

    d = distances.copy()
    d[~finite_mask] = fallback
    d[d < 0.0] = 0.0
    np.fill_diagonal(d, np.inf)

    nearest_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest_indices = np.argpartition(
        d, kth=nearest_count - 1, axis=1
    )[:, :nearest_count]

    positive_values = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    d /= scale
    d += 1.0
    np.reciprocal(d, out=d)
    d[~np.isfinite(d)] = 0.0

    score = d
    score *= 0.55 / 0.75

    nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    nearest[rows, nearest_indices] = 1.0
    np.fill_diagonal(nearest, 0.0)

    mutual = nearest * nearest.T
    score += (0.05 / 0.75) * (nearest + nearest.T)
    score += (0.10 / 0.75) * mutual

    score = (score + score.T) / 2.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
