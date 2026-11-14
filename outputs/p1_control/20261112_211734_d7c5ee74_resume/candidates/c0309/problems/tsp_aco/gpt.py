import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, locally scaled edge-priority matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(valid, distances, fallback)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    median_positive = float(np.median(positive))
    floor = max(median_positive * 1e-10, np.finfo(float).eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), floor)
    inv_local = 1.0 / local_scale

    if n > 2:
        second = np.partition(d, kth=1, axis=1)[:, 1]
    else:
        second = d[:, 0]
    second = np.where(np.isfinite(second), second, local_scale)
    second = np.maximum(second, floor)

    inv_pair = 1.0 / np.sqrt(local_scale[:, None] * local_scale[None, :])
    inv_candidate = 2.0 / (
        local_scale[:, None] + local_scale[None, :] + floor
    )

    exponent = d * (
        inv_pair
        + 0.5 * inv_local[:, None]
        + 0.5 * inv_local[None, :]
        + inv_candidate
    )
    score = np.exp(-np.clip(exponent, 0.0, 120.0))

    alternative = np.sqrt(second[:, None] * second[None, :])
    margin = alternative / (d + alternative + floor)
    margin = np.clip(margin, 0.0, 1.0)

    rank_support = (
        local_scale[:, None] / (d + local_scale[:, None] + floor)
        + local_scale[None, :] / (d + local_scale[None, :] + floor)
    )
    rank_support = np.clip(0.5 * rank_support, 0.0, 1.0)

    score *= 1.0 + 0.05 * margin + 0.04 * rank_support
    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
