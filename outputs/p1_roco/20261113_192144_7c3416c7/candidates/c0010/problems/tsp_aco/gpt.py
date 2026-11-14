import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    finite_edges = np.isfinite(work)
    positive = work[finite_edges & (work > 0)]

    if positive.size == 0:
        return np.zeros_like(work, dtype=float)

    scale = float(np.median(positive))
    scale = max(scale, np.finfo(float).eps)

    # Robust inverse-distance preference.
    inverse = 1.0 / (1.0 + work / scale)
    inverse[~finite_edges] = 0.0

    # Prefer edges incident to locally sparse nodes.
    k = max(1, min(3, n - 1))
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale /= max(float(np.mean(local_scale)), np.finfo(float).eps)

    sparsity = np.sqrt(local_scale[:, None] * local_scale[None, :])
    sparsity = np.clip(sparsity, 0.5, 2.0)

    # Detect mutual nearest-neighbor edges.
    nearest_neighbor = np.argmin(work, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_neighbor[i]
        if nearest_neighbor[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score = inverse * (0.72 + 0.18 * sparsity) + 0.10 * mutual
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
