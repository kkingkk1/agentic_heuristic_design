import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric multiscale graph-diffusion prior for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0]

    if positive_values.size:
        scales = np.quantile(
            positive_values,
            [0.25, 0.50, 0.75],
        )
        fallback = float(np.median(positive_values))
    else:
        scales = np.ones(3, dtype=float)
        fallback = 1.0

    fallback = max(fallback, np.finfo(float).eps)
    scales = np.where(
        np.isfinite(scales) & (scales > 0),
        scales,
        fallback,
    )

    safe_d = np.where(valid, d, 0.0)
    symmetric_d = 0.5 * (safe_d + safe_d.T)
    symmetric_valid = valid | valid.T
    symmetric_d = np.where(symmetric_valid, symmetric_d, 0.0)
    np.fill_diagonal(symmetric_d, 0.0)

    direct = np.zeros((n, n), dtype=float)
    diffused = np.zeros((n, n), dtype=float)

    for scale in scales:
        affinity = np.exp(-np.square(symmetric_d / scale))
        affinity *= symmetric_valid
        np.fill_diagonal(affinity, 0.0)

        degree = affinity.sum(axis=1)
        inv_sqrt_degree = np.zeros(n, dtype=float)
        positive_degree = degree > 0
        inv_sqrt_degree[positive_degree] = 1.0 / np.sqrt(
            degree[positive_degree]
        )

        normalized = (
            inv_sqrt_degree[:, None]
            * affinity
            * inv_sqrt_degree[None, :]
        )

        direct += affinity
        diffused += normalized @ normalized

    direct /= float(len(scales))
    diffused /= float(len(scales))

    direct_max = np.max(direct)
    if direct_max > 0:
        direct /= direct_max

    diffused *= symmetric_valid
    diffused_max = np.max(diffused)
    if diffused_max > 0:
        diffused /= diffused_max

    score = 0.60 * direct + 0.40 * diffused
    score *= symmetric_valid
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
