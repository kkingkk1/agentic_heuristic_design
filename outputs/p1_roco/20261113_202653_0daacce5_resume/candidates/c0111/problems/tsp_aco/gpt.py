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
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    nonnegative = np.maximum(distances, 0.0)
    positive = nonnegative[valid & (nonnegative > 0.0)]

    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).tiny)
    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)

    safe_distances = np.where(valid, nonnegative, np.inf)

    inverse_distance = 1.0 / (safe_distances + eps)
    inverse_distance[~valid] = 0.0

    robust_closeness = np.log1p(scale * inverse_distance)
    robust_closeness[~valid] = 0.0

    row_max = np.max(robust_closeness, axis=1, keepdims=True)
    col_max = np.max(robust_closeness, axis=0, keepdims=True)

    row_competitiveness = robust_closeness / (row_max + eps)
    col_competitiveness = robust_closeness / (col_max + eps)

    mutual_competitiveness = np.sqrt(
        np.clip(row_competitiveness * col_competitiveness, 0.0, None)
    )

    priors = robust_closeness * (
        0.40
        + 0.25 * row_competitiveness
        + 0.20 * col_competitiveness
        + 0.15 * mutual_competitiveness
    )

    priors[~valid] = 0.0

    maximum = float(np.max(priors))
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    return priors
