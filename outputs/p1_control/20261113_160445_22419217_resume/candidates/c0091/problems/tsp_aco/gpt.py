import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric entropy-balanced prior for TSP edge sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)
    directed_valid = finite_nonnegative & off_diagonal

    # Build a symmetric undirected distance matrix from available directions.
    forward = np.where(directed_valid, raw, np.nan)
    backward = np.where(directed_valid.T, raw.T, np.nan)

    pair_count = np.isfinite(forward).astype(float) + np.isfinite(backward).astype(float)
    pair_sum = np.nansum(np.stack((forward, backward)), axis=0)

    valid = pair_count > 0.0
    distances = np.zeros((n, n), dtype=float)
    distances[valid] = pair_sum[valid] / pair_count[valid]
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive = distances[valid & (distances > 0.0)]
    if positive.size:
        center = float(np.median(positive))
        deviation = float(np.median(np.abs(positive - center)))
        temperature = max(center + deviation, np.finfo(float).eps)
    else:
        temperature = 1.0

    # Entropy-regularized preference kernel.
    finite_distances = np.where(valid, distances, 0.0)
    shifted = np.maximum(finite_distances, 0.0)
    kernel = np.exp(-np.minimum(shifted / temperature, 60.0))
    kernel[~valid] = 0.0

    # Sinkhorn balancing makes each node compete for a shared probability mass.
    balanced = kernel.copy()
    for _ in range(120):
        row_sum = balanced.sum(axis=1)
        balanced = np.divide(
            balanced,
            row_sum[:, None],
            out=np.zeros_like(balanced),
            where=row_sum[:, None] > 0.0,
        )

        col_sum = balanced.sum(axis=0)
        balanced = np.divide(
            balanced,
            col_sum[None, :],
            out=np.zeros_like(balanced),
            where=col_sum[None, :] > 0.0,
        )

    # Require mutual support in the balanced assignment prior.
    score = np.sqrt(np.maximum(balanced, 0.0) * np.maximum(balanced.T, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
