import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return locally scaled, symmetric edge priorities for stochastic TSP sampling."""
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

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(valid, distances, fallback)
    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    scale_floor = max(
        float(np.median(positive)) * 1e-10,
        np.finfo(float).eps,
    )

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), scale_floor)

    if n > 2:
        alternative = np.partition(d, kth=1, axis=1)[:, 1]
    else:
        alternative = d[:, 0]

    alternative = np.where(
        np.isfinite(alternative),
        alternative,
        local_scale,
    )
    alternative = np.maximum(alternative, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = d / (pair_scale + scale_floor)
    normalized_distance = np.clip(normalized_distance, 0.0, 40.0)

    distance_affinity = np.exp(-normalized_distance)

    endpoint_i = np.exp(
        -np.clip(
            d / (local_scale[:, None] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            d / (local_scale[None, :] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_support = np.exp(
        -0.5 * (ranks + ranks.T) / float(max(k, 1))
    )

    adaptive_floor = 0.06 + 0.34 * np.exp(
        -0.75 * normalized_distance
    )
    adaptive_floor = np.clip(adaptive_floor, 0.06, 0.40)

    candidate_support = (
        adaptive_floor
        + (1.0 - adaptive_floor) * rank_support
    )

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    score = (
        distance_affinity
        * endpoint_compatibility
        * candidate_support
        * (1.0 + 0.05 * margin)
    )

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
