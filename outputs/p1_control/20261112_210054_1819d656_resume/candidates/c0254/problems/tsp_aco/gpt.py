import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    d = np.where(valid, distances, fallback)
    d = 0.5 * (d + d.T)
    symmetric_valid = valid & valid.T
    np.fill_diagonal(d, np.inf)

    k = min(n - 1, 8)
    local = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(local, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = d / (pair_scale + np.finfo(float).eps)

    score = 1.0 / (1.0 + relative)
    score *= 0.92 + 0.08 / (1.0 + d / (scale + np.finfo(float).eps))

    endpoint_balance = np.sqrt(
        (local_scale[:, None] + np.finfo(float).eps)
        / (local_scale[None, :] + np.finfo(float).eps)
    )
    endpoint_balance = 0.5 * (endpoint_balance + 1.0 / endpoint_balance)
    score /= np.minimum(endpoint_balance, 2.0)

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
