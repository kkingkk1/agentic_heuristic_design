import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return sparse, symmetric local-neighborhood priorities for TSP sampling."""
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

    k = max(1, min(n - 1, int(np.sqrt(n))))
    nearest = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(work, nearest, axis=1)

    eps = np.finfo(float).eps
    scale_floor = max(float(np.median(positive)) * 1e-10, eps)
    local_scale = np.maximum(
        np.mean(nearest_distances, axis=1),
        scale_floor,
    )

    candidate = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate[rows, nearest] = True

    mutual = candidate & candidate.T
    supported = candidate | candidate.T

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = work / (pair_scale + scale_floor)

    distance_score = 1.0 / (1.0 + relative_distance)

    endpoint_support = 0.5 * (
        local_scale[:, None] / (local_scale[:, None] + work + scale_floor)
        + local_scale[None, :] / (local_scale[None, :] + work + scale_floor)
    )

    alternative = np.partition(work, kth=min(1, n - 2), axis=1)[:, min(1, n - 2)]
    alternative = np.maximum(alternative, scale_floor)

    detour_margin = (
        np.sqrt(alternative[:, None] * alternative[None, :]) - work
    ) / (
        np.sqrt(alternative[:, None] * alternative[None, :])
        + work
        + scale_floor
    )
    detour_margin = np.clip(detour_margin, 0.0, 1.0)

    score = distance_score * (0.75 + 0.25 * endpoint_support)
    score *= 1.0 + 0.10 * detour_margin
    score *= np.where(mutual, 1.12, 1.0)
    score *= supported

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

    return score.astype(float, copy=False)
