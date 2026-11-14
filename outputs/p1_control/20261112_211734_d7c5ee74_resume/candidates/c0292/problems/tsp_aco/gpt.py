import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized local-structure-aware edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)
    symmetric_valid = valid & valid.T

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[symmetric_valid & (distances > 0.0)]

    if positive.size == 0:
        score = symmetric_valid.astype(float)
        np.fill_diagonal(score, 0.0)
        maximum = float(np.max(score))
        return score / maximum if maximum > 0.0 else score

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(valid, distances, fallback)
    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive_median = float(np.median(positive))
    floor = max(
        positive_median * 1e-10,
        np.finfo(float).eps,
    )

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, floor)

    ordered = np.sort(d, axis=1)
    nearest_alternative = (
        ordered[:, 1] if n > 2 else ordered[:, 0]
    )
    nearest_alternative = np.where(
        np.isfinite(nearest_alternative),
        nearest_alternative,
        local_scale,
    )
    nearest_alternative = np.maximum(nearest_alternative, floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = d / (pair_scale + floor)

    distance_affinity = np.exp(
        -np.clip(normalized_distance, 0.0, 40.0)
    )

    endpoint_ratio_i = d / (local_scale[:, None] + floor)
    endpoint_ratio_j = d / (local_scale[None, :] + floor)

    endpoint_compatibility = 1.0 / np.sqrt(
        (1.0 + np.clip(endpoint_ratio_i, 0.0, 40.0))
        * (1.0 + np.clip(endpoint_ratio_j, 0.0, 40.0))
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    rank_support = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )

    alternative_scale = np.sqrt(
        nearest_alternative[:, None]
        * nearest_alternative[None, :]
    )

    regret_i = np.clip(
        (nearest_alternative[:, None] - d)
        / (nearest_alternative[:, None] + floor),
        0.0,
        1.0,
    )
    regret_j = np.clip(
        (nearest_alternative[None, :] - d)
        / (nearest_alternative[None, :] + floor),
        0.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.12 * np.sqrt(regret_i * regret_j)

    score = (
        distance_affinity
        * endpoint_compatibility
        * rank_support
        * regret_bonus
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
