import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge-priority indicators for stochastic TSP sampling."""
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
    np.fill_diagonal(d, np.inf)

    median_distance = float(np.median(positive))
    floor = max(median_distance * 1e-10, np.finfo(float).eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, floor)

    if n > 2:
        alternative = np.partition(d, kth=1, axis=1)[:, 1]
    else:
        alternative = nearest[:, 0]

    alternative = np.where(
        np.isfinite(alternative),
        alternative,
        local_scale,
    )
    alternative = np.maximum(alternative, floor)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized = d / (pair_scale + floor)

    attraction = np.exp(-np.clip(normalized, 0.0, 40.0))

    endpoint_ratio_i = d / (local_scale[:, None] + floor)
    endpoint_ratio_j = d / (local_scale[None, :] + floor)
    compatibility = 1.0 / np.sqrt(
        (1.0 + endpoint_ratio_i) * (1.0 + endpoint_ratio_j)
    )

    kth_distance = nearest[:, -1]
    candidate_i = d <= (kth_distance[:, None] + floor)
    candidate_j = d <= (kth_distance[None, :] + floor)

    candidate_support = (
        1.0
        + 0.18 * candidate_i.astype(float)
        + 0.18 * candidate_j.astype(float)
        + 0.14 * (candidate_i & candidate_j).astype(float)
    )

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + floor
    )
    margin_bonus = 1.0 + 0.06 * np.clip(margin, 0.0, 1.0)

    score = (
        attraction
        * compatibility
        * candidate_support
        * margin_bonus
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
