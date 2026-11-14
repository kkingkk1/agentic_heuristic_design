import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = distance_matrix.copy()
    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    positive = d[valid & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Symmetrize because an undirected TSP edge should have one shared prior.
    ds = np.where(valid, d, np.nan)
    ds = np.nanmean(np.stack((ds, ds.T)), axis=0)
    ds = np.where(np.isfinite(ds), ds, np.inf)
    np.fill_diagonal(ds, np.inf)

    # Robust inverse-distance preference.
    inverse = 1.0 / (1.0 + ds / scale)
    inverse[~np.isfinite(inverse)] = 0.0
    inverse = 0.5 * (inverse + inverse.T)

    # Rank-based neighborhood preference.
    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(ds[i], kind="stable")
        rank = 0
        for j in order:
            if not np.isfinite(ds[i, j]):
                continue
            ranks[i, j] = 1.0 / (1.0 + rank)
            rank += 1

    neighborhood = 0.5 * (ranks + ranks.T)

    # Mutual-nearest-neighbor bonus.
    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        finite_indices = np.flatnonzero(np.isfinite(ds[i]))
        if finite_indices.size:
            nearest[i] = finite_indices[np.argmin(ds[i, finite_indices])]

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score = 0.60 * inverse + 0.30 * neighborhood + 0.10 * mutual
    score[~np.isfinite(ds)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
