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
    positive = distances[finite & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    eps = np.sqrt(np.finfo(float).eps) * max(scale, np.finfo(float).tiny)

    valid_distances = np.maximum(distances, 0.0)
    valid_distances[~finite] = np.inf
    np.fill_diagonal(valid_distances, np.inf)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inverse = 1.0 / (valid_distances + eps)

    inverse[~finite] = 0.0

    row_peak = np.max(inverse, axis=1, keepdims=True)
    col_peak = np.max(inverse, axis=0, keepdims=True)

    row_strength = np.divide(
        inverse,
        row_peak + eps,
        out=np.zeros_like(inverse),
        where=row_peak > 0.0,
    )
    col_strength = np.divide(
        inverse,
        col_peak + eps,
        out=np.zeros_like(inverse),
        where=col_peak > 0.0,
    )

    global_peak = float(np.max(inverse))
    global_strength = inverse / (global_peak + eps)

    compressed = np.log1p(inverse * scale) / np.log1p(global_peak * scale + eps)

    priors = compressed * (
        0.42
        + 0.30 * row_strength
        + 0.20 * col_strength
        + 0.08 * global_strength
    )

    priors[~finite] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    return priors
