import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-inclusion priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distances.copy()
    finite_off_diagonal = np.isfinite(d)
    np.fill_diagonal(finite_off_diagonal, False)

    if not np.any(finite_off_diagonal):
        return np.zeros_like(d, dtype=float)

    finite_values = d[finite_off_diagonal]
    fallback = float(np.max(finite_values)) if finite_values.size else 1.0
    d[~np.isfinite(d)] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive_values = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Smooth inverse-distance attraction.
    inverse_distance = 1.0 / (1.0 + d / scale)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    # Row-wise nearest-neighbor rank preference.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0

    # Mutual-neighborhood compatibility.
    nearest_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.zeros((n, n), dtype=float)
    nearest[row_indices, order[:, :nearest_count]] = 1.0
    np.fill_diagonal(nearest, 0.0)
    mutual = nearest * nearest.T
    one_sided = nearest + nearest.T - mutual

    score = (
        0.55 * inverse_distance
        + 0.25 * (rank_score + rank_score.T) / 2.0
        + 0.15 * mutual
        + 0.05 * one_sided
    )

    score = (score + score.T) / 2.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
