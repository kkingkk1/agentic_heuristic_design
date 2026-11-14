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
    nonnegative = np.maximum(np.where(finite, distances, 0.0), 0.0)
    symmetric = 0.5 * (nonnegative + nonnegative.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & finite.T & off_diagonal

    positive = symmetric[valid & (symmetric > 0.0)]
    if positive.size:
        fallback = float(np.median(positive))
    else:
        fallback = 1.0
    fallback = max(fallback, np.finfo(float).eps)

    work = np.where(valid, symmetric, fallback)
    np.fill_diagonal(work, np.inf)

    finite_work = np.where(np.isfinite(work), work, fallback)
    neighbor_count = max(1, min(n - 1, int(np.sqrt(n)) + 1))

    ordered = np.sort(finite_work, axis=1)
    local_bandwidth = np.mean(ordered[:, :neighbor_count], axis=1)
    local_bandwidth = np.maximum(local_bandwidth, np.finfo(float).eps)

    scale = np.sqrt(
        local_bandwidth[:, None] * local_bandwidth[None, :]
    )
    affinity = np.exp(-np.square(finite_work / scale))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    degree = np.sum(affinity, axis=1)
    degree = np.maximum(degree, np.finfo(float).eps)

    normalized = affinity / np.sqrt(degree[:, None] * degree[None, :])
    diffusion = normalized @ normalized
    diffusion = np.maximum(diffusion, 0.0)
    np.fill_diagonal(diffusion, 0.0)

    affinity_max = np.max(affinity, axis=1, keepdims=True)
    affinity_max = np.maximum(affinity_max, np.finfo(float).eps)
    direct = affinity / affinity_max

    diffusion_max = np.max(diffusion, axis=1, keepdims=True)
    diffusion_max = np.maximum(diffusion_max, np.finfo(float).eps)
    shared_support = diffusion / diffusion_max

    score = 0.30 * direct + 0.70 * shared_support
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
