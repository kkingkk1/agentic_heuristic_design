import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    # Use a symmetric, nonnegative working matrix and suppress self-loops.
    d = np.maximum(0.5 * (d + d.T), 0.0)
    valid = ~np.eye(n, dtype=bool)
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[valid & (work > 0) & np.isfinite(work)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    eps = np.finfo(float).eps * max(scale, 1.0)

    # Short-edge baseline, bounded for zero-distance edges.
    inverse = np.zeros((n, n), dtype=float)
    finite = np.isfinite(work)
    inverse[finite] = 1.0 / (work[finite] + eps)
    inverse /= max(float(inverse.max()), eps)

    # Reciprocal rank in each endpoint's neighborhood.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        rank = 1.0
        for j in order[i]:
            if np.isfinite(work[i, j]):
                ranks[i, j] = 1.0 / rank
                rank += 1.0
    mutual_rank = np.sqrt(ranks * ranks.T)

    # Soft local contrast: reward edges short relative to both endpoint scales.
    local_scale = np.zeros(n, dtype=float)
    for i in range(n):
        values = work[i, np.isfinite(work[i])]
        if values.size:
            local_scale[i] = np.median(values)
    local_scale[local_scale <= 0] = scale

    contrast = np.zeros((n, n), dtype=float)
    denominator = np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps
    contrast[finite] = np.exp(-work[finite] / denominator[finite])

    # Endpoint scarcity and mutual neighborhood support.
    local_mass = contrast.sum(axis=1)
    scarcity = 1.0 / np.maximum(local_mass, eps)
    scarcity /= max(float(scarcity.mean()), eps)
    endpoint_support = np.sqrt(scarcity[:, None] * scarcity[None, :])

    neighborhood_support = contrast / np.maximum(local_mass[:, None], eps)
    mutual_support = np.sqrt(neighborhood_support * neighborhood_support.T)

    score = (
        0.50 * inverse
        + 0.20 * mutual_rank
        + 0.15 * contrast
        + 0.10 * mutual_support
        + 0.05 * np.clip(endpoint_support, 0.0, 2.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0

    maximum = float(score.max(initial=0.0))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
