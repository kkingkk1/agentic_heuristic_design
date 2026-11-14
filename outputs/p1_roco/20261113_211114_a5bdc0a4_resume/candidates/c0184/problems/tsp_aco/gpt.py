import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    if not np.any(finite):
        return np.zeros_like(distances, dtype=float)

    positive = distances[finite & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)

    safe_distances = np.where(
        finite,
        np.maximum(distances, 0.0),
        np.inf,
    )
    np.fill_diagonal(safe_distances, np.inf)

    inverse_distance = 1.0 / (safe_distances + eps)
    inverse_distance[~finite] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    col_max = np.max(inverse_distance, axis=0, keepdims=True)

    row_competitiveness = inverse_distance / (row_max + eps)
    col_competitiveness = inverse_distance / (col_max + eps)

    priors = inverse_distance * (
        0.42
        + 0.34 * row_competitiveness
        + 0.24 * col_competitiveness
    )

    priors[~finite] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = np.max(priors)
    if maximum > 0.0:
        priors /= maximum

    return priors
