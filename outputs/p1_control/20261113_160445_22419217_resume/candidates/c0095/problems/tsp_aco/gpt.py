import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally connectivity-aware edge prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, 0.0)

    conductance = np.zeros_like(d, dtype=float)
    conductance[valid] = 1.0 / (1.0 + safe[valid] / scale)
    conductance = 0.5 * (conductance + conductance.T)
    conductance[~valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    degree = np.sum(conductance, axis=1)
    laplacian = -conductance
    np.fill_diagonal(laplacian, degree)

    try:
        laplacian_pinv = np.linalg.pinv(
            laplacian,
            hermitian=True,
            rcond=1e-12,
        )
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian, rcond=1e-12)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance = 0.5 * (resistance + resistance.T)

    score = conductance * resistance
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
