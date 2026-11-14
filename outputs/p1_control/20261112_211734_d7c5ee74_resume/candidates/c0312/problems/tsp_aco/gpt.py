import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric locally scaled edge-priority matrix."""
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

    scale = float(np.median(positive))
    floor = max(scale * 1e-10, np.finfo(float).eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), floor)
    inv_scale = 1.0 / local_scale

    pair_inv_scale = 2.0 * inv_scale[:, None] * inv_scale[None, :] / (
        inv_scale[:, None] + inv_scale[None, :] + floor
    )
    exponent = d * pair_inv_scale
    score = np.exp(-np.clip(exponent, 0.0, 80.0))

    relative_shortness = 0.5 * (
        local_scale[:, None] / (d + local_scale[:, None] + floor)
        + local_scale[None, :] / (d + local_scale[None, :] + floor)
    )
    score *= 1.0 + 0.02 * np.clip(relative_shortness, 0.0, 1.0)

    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
