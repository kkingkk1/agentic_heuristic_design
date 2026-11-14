import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0)

    if not np.any(positive):
        return np.zeros_like(distances, dtype=float)

    finite_values = distances[positive]
    fallback = float(np.max(finite_values) * 10.0 + 1.0)

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, fallback)

    valid = np.isfinite(d) & (d > 0)
    valid_values = d[valid]
    scale = float(np.median(valid_values)) if valid_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Smooth preference for short edges.
    short_edge = np.exp(-d / scale)

    # Estimate how isolated each node is from its nearest neighbors.
    neighbor_distances = d.copy()
    np.fill_diagonal(neighbor_distances, np.inf)
    k = max(1, min(5, n - 1))
    nearest = np.partition(neighbor_distances, kth=k - 1, axis=1)[:, :k]
    isolation = np.mean(nearest, axis=1)
    isolation = isolation / (np.mean(isolation) + np.finfo(float).eps)
    isolation_factor = 1.0 + 0.25 * np.minimum(isolation[:, None] + isolation[None, :], 4.0)

    # Mutual-neighborhood compatibility.
    ranks = np.argsort(np.argsort(neighbor_distances, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    mutuality = np.sqrt(rank_score * rank_score.T)

    score = short_edge * isolation_factor * (0.65 + 0.35 * mutuality)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
