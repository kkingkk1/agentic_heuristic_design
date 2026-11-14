import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        finite_values = distances[finite]
        nonnegative = finite_values[finite_values >= 0.0]
        replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    nearest = np.min(distances, axis=1)
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        scale,
    )
    nearest = np.maximum(nearest, np.finfo(float).eps)

    distance_signal = scale / (distances + scale)
    distance_signal[~np.isfinite(distance_signal)] = 0.0

    local_scale = 0.5 * (
        1.0 / nearest[:, None] + 1.0 / nearest[None, :]
    )
    local_signal = 1.0 / (1.0 + distances * local_scale)
    local_signal[~np.isfinite(local_signal)] = 0.0

    score = 0.75 * distance_signal + 0.25 * local_signal
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
