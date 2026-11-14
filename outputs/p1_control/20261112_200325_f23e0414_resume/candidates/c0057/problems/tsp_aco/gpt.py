import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(d) & (d > 0)
    valid[np.diag_indices(n)] = False

    positive = d[valid]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)
    finite_rows = np.any(valid, axis=1)

    row_min = np.min(safe, axis=1, keepdims=True)
    row_min = np.where(np.isfinite(row_min), row_min, scale)

    relative_distance = np.maximum(safe - row_min, 0.0) / scale

    inverse = np.where(valid, 1.0 / (d + scale * 1e-9), 0.0)
    closeness = np.where(valid, np.exp(-relative_distance), 0.0)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    column_ranks = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = column_ranks
    ranks[~valid] = float(n)
    rank_score = np.where(valid, 1.0 / (1.0 + ranks), 0.0)

    nearest = np.argmin(safe, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    mutual[rows[finite_rows], nearest[finite_rows]] = 1.0
    mutual = mutual * mutual.T
    mutual[~valid] = 0.0

    def normalize_rows(values: np.ndarray) -> np.ndarray:
        maxima = np.max(values, axis=1, keepdims=True)
        return np.divide(
            values,
            maxima,
            out=np.zeros_like(values, dtype=float),
            where=maxima > 0,
        )

    score = (
        0.48 * normalize_rows(inverse)
        + 0.27 * normalize_rows(closeness)
        + 0.15 * normalize_rows(rank_score)
        + 0.10 * mutual
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
