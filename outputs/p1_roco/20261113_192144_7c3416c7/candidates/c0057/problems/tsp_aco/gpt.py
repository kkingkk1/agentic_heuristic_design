import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    safe_distances = np.where(finite, np.maximum(distances, 0.0), 0.0)
    np.fill_diagonal(safe_distances, 0.0)

    valid_counts = valid.sum(axis=1)
    row_sums = np.sum(np.where(valid, safe_distances, 0.0), axis=1)
    col_counts = valid.sum(axis=0)
    col_sums = np.sum(np.where(valid, safe_distances, 0.0), axis=0)

    positive = safe_distances[valid & (safe_distances > 0.0)]
    global_scale = float(np.mean(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    row_scale = np.divide(
        row_sums,
        valid_counts,
        out=np.full(n, global_scale, dtype=float),
        where=valid_counts > 0,
    )
    col_scale = np.divide(
        col_sums,
        col_counts,
        out=np.full(n, global_scale, dtype=float),
        where=col_counts > 0,
    )

    row_scale = np.maximum(row_scale, global_scale * 1e-12)
    col_scale = np.maximum(col_scale, global_scale * 1e-12)

    row_attraction = row_scale[:, None] / (
        row_scale[:, None] + safe_distances
    )
    col_attraction = col_scale[None, :] / (
        col_scale[None, :] + safe_distances
    )

    priors = np.sqrt(row_attraction * col_attraction)
    priors[~valid] = 0.0

    maximum = np.max(priors)
    if maximum > 0.0:
        priors /= maximum

    return priors
