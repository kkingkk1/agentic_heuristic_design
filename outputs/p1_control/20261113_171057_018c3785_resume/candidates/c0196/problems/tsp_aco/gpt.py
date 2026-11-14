import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced two-sided Gibbs edge priors for a directed TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    affinity = np.zeros((n, n), dtype=float)

    if n <= 1:
        return affinity

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return affinity

    safe_d = np.where(valid, d, np.inf)
    row_min = np.min(safe_d, axis=1)
    col_min = np.min(safe_d, axis=0)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    temperature = max(0.30 * scale, np.finfo(float).eps)

    col_excess = np.zeros_like(d, dtype=float)

    np.subtract(d, row_min[:, None], out=affinity, where=valid)
    np.maximum(affinity, 0.0, out=affinity)

    np.subtract(d, col_min[None, :], out=col_excess, where=valid)
    np.maximum(col_excess, 0.0, out=col_excess)

    np.add(affinity, col_excess, out=affinity)
    affinity *= -0.5 / temperature
    np.exp(affinity, out=affinity, where=valid)
    affinity[~valid] = 0.0

    for _ in range(4):
        row_sum = affinity.sum(axis=1)
        np.divide(
            affinity,
            row_sum[:, None],
            out=affinity,
            where=row_sum[:, None] > 0.0,
        )

        col_sum = affinity.sum(axis=0)
        np.divide(
            affinity,
            col_sum[None, :],
            out=affinity,
            where=col_sum[None, :] > 0.0,
        )

    maximum = float(np.max(affinity))
    if maximum > 0.0 and np.isfinite(maximum):
        affinity /= maximum

    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    return affinity
