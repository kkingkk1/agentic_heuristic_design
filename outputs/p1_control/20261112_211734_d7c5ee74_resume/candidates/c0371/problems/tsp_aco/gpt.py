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

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    work = np.where(valid, distances, fallback)
    work = np.maximum(work, 0.0)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    scale_floor = max(float(np.median(positive)) * 1e-10, eps)

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        scale_floor,
    )

    if n > 2:
        alternatives = np.partition(work, kth=1, axis=1)[:, 1]
    else:
        alternatives = nearest[:, 0]

    alternatives = np.where(
        np.isfinite(alternatives),
        alternatives,
        local_scale,
    )
    alternatives = np.maximum(alternatives, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = np.clip(
        work / (pair_scale + scale_floor),
        0.0,
        40.0,
    )

    endpoint_ratio = np.clip(
        work / (local_scale[:, None] + scale_floor)
        + work / (local_scale[None, :] + scale_floor),
        0.0,
        80.0,
    )

    exponent = (
        1.25 * normalized_distance
        + 0.45 * endpoint_ratio
    )

    score = np.exp(-np.clip(exponent, 0.0, 80.0))

    alternative_scale = np.sqrt(
        alternatives[:, None] * alternatives[None, :]
    )
    margin = (
        alternative_scale - work
    ) / (
        alternative_scale + work + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    score *= 1.0 + 0.08 * margin

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
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
