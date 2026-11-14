import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-priority indicators for stochastic TSP sampling."""
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

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    floor = max(scale * 1e-12, np.finfo(float).eps)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    k = max(1, min(n - 1, int(np.ceil(0.65 * np.sqrt(n)))))

    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, floor)

    nearest = ordered[:, 0]
    if n > 2:
        second_nearest = ordered[:, 1]
    else:
        second_nearest = nearest

    nearest = np.maximum(np.where(np.isfinite(nearest), nearest, local_scale), floor)
    second_nearest = np.maximum(
        np.where(np.isfinite(second_nearest), second_nearest, local_scale),
        floor,
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized = d / (pair_scale + floor)

    distance_affinity = np.exp(
        -np.clip(1.25 * normalized, 0.0, 50.0)
    )

    reciprocal_affinity = 1.0 / (
        1.0 + np.sqrt(d / (pair_scale + floor))
    )

    endpoint_i = np.exp(
        -np.clip(0.82 * d / (local_scale[:, None] + floor), 0.0, 50.0)
    )
    endpoint_j = np.exp(
        -np.clip(0.82 * d / (local_scale[None, :] + floor), 0.0, 50.0)
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(0.70 * k, 1.0)
    rank_support = np.exp(
        -np.clip((ranks + ranks.T) / (2.0 * rank_scale), 0.0, 50.0)
    )

    candidate_threshold = np.minimum(
        second_nearest[:, None],
        second_nearest[None, :],
    )
    local_competition = candidate_threshold / (
        d + candidate_threshold + floor
    )
    local_competition = np.clip(local_competition, 0.0, 1.0)

    node_regret = np.maximum(second_nearest - nearest, 0.0) / (
        second_nearest + floor
    )
    regret_support = 1.0 + 0.10 * np.sqrt(
        node_regret[:, None] * node_regret[None, :]
    )

    score = (
        distance_affinity
        * reciprocal_affinity
        * endpoint_compatibility
        * rank_support
        * (1.0 + 0.07 * local_competition)
        * regret_support
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
