import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return two-sided balanced edge priors for a directed TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    safe_d = np.where(valid, d, np.inf)
    row_min = np.min(safe_d, axis=1)
    col_min = np.min(safe_d, axis=0)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    temperature = max(0.30 * scale, np.finfo(float).eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)
    np.subtract(d, row_min[:, None], out=row_excess, where=valid)
    np.subtract(d, col_min[None, :], out=col_excess, where=valid)
    np.maximum(row_excess, 0.0, out=row_excess)
    np.maximum(col_excess, 0.0, out=col_excess)

    affinity = np.zeros_like(d)
    exponent = -0.5 * (row_excess + col_excess) / temperature
    np.exp(exponent, out=affinity, where=valid)
    affinity[~valid] = 0.0

    for _ in range(4):
        row_sum = affinity.sum(axis=1)
        row_scale = np.divide(
            1.0,
            row_sum,
            out=np.zeros_like(row_sum),
            where=row_sum > 0.0,
        )
        affinity *= row_scale[:, None]

        col_sum = affinity.sum(axis=0)
        col_scale = np.divide(
            1.0,
            col_sum,
            out=np.zeros_like(col_sum),
            where=col_sum > 0.0,
        )
        affinity *= col_scale[None, :]

    maximum = float(np.max(affinity))
    if maximum > 0.0 and np.isfinite(maximum):
        affinity /= maximum

    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0
    return affinity
