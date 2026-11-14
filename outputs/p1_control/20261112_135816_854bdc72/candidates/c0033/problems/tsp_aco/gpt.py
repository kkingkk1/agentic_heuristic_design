import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global connectivity-based edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & off_diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    d = np.maximum(distances, 0.0)
    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & off_diagonal

    positive = d[valid & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    conductance = np.zeros((n, n), dtype=float)
    normalized = np.zeros_like(d, dtype=float)
    normalized[valid] = d[valid] / scale
    conductance[valid] = np.exp(-np.clip(normalized[valid], 0.0, 50.0))
    conductance = 0.5 * (conductance + conductance.T)

    degree = np.sum(conductance, axis=1)
    laplacian = np.diag(degree) - conductance

    try:
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)

    score = conductance * resistance
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
