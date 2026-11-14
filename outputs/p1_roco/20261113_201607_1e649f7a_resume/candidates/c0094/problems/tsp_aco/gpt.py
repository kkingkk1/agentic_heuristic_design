import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric degree-balanced prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    nonnegative = finite & (d >= 0.0)
    valid = nonnegative & nonnegative.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros((n, n), dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])
    np.fill_diagonal(symmetric_d, 0.0)

    values = symmetric_d[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    temperature = max(0.35 * scale, 0.5 * spread, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.clip(symmetric_d[valid] / temperature, 0.0, 80.0)
    )

    degree = affinity.sum(axis=1)
    if not np.any(degree > 0.0):
        return np.zeros((n, n), dtype=float)

    weights = affinity.copy()
    target_degree = 2.0

    for _ in range(80):
        degree = weights.sum(axis=1)
        safe_degree = np.maximum(degree, eps)

        correction = np.sqrt(target_degree / safe_degree)
        weights *= correction[:, None]
        weights *= correction[None, :]
        weights[~valid] = 0.0

        if np.max(np.abs(weights.sum(axis=1) - target_degree)) < 1e-7:
            break

    degree = weights.sum(axis=1)
    safe_degree = np.maximum(degree, eps)

    residual = target_degree / safe_degree
    residual = np.sqrt(np.maximum(residual[:, None] * residual[None, :], 0.0))

    score = weights * residual
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
