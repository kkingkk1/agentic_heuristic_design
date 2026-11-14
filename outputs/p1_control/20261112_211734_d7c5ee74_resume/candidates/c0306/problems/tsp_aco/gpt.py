import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric endpoint-aware edge-priority matrix."""
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
    scale_floor = max(
        median_positive * 1e-10,
        np.finfo(float).eps,
    )

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest_values = np.partition(
        d,
        kth=k - 1,
        axis=1,
    )[:, :k]
    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    if n > 2:
        second_nearest = np.partition(d, kth=1, axis=1)[:, 1]
    else:
        second_nearest = np.partition(d, kth=0, axis=1)[:, 0]

    second_nearest = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        local_scale,
    )
    second_nearest = np.maximum(second_nearest, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    candidate_scale = 0.5 * (
        local_scale[:, None] + local_scale[None, :]
    )

    # Combine the three monotonic distance penalties into one exponential.
    exponent = (
        d / (pair_scale + scale_floor)
        + 0.5 * d / (local_scale[:, None] + scale_floor)
        + 0.5 * d / (local_scale[None, :] + scale_floor)
        + d / (candidate_scale + scale_floor)
    )

    score = np.exp(-np.clip(exponent, 0.0, 120.0))

    alternative_scale = np.sqrt(
        second_nearest[:, None] * second_nearest[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    score *= 1.0 + 0.06 * margin
    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
