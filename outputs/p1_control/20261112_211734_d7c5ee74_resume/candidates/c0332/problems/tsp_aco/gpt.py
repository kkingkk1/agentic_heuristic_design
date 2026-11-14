import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & off_diagonal
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
    d = np.maximum((d + d.T) * 0.5, 0.0)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive_scale = max(float(np.median(positive)), eps)
    scale_floor = max(positive_scale * 1e-10, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    local_values = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(local_values.mean(axis=1), scale_floor)

    if n > 2:
        alternatives = np.partition(d, kth=1, axis=1)[:, 1]
    else:
        alternatives = local_values[:, 0]

    alternatives = np.where(
        np.isfinite(alternatives),
        alternatives,
        local_scale,
    )
    alternatives = np.maximum(alternatives, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / (pair_scale + eps)
    endpoint_i = d / (local_scale[:, None] + eps)
    endpoint_j = d / (local_scale[None, :] + eps)

    distance_affinity = np.exp(
        -np.clip(relative_distance ** 0.95, 0.0, 50.0)
    )

    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(endpoint_i, 0.0, 50.0))
        * np.exp(-np.clip(endpoint_j, 0.0, 50.0))
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    rank_affinity = np.exp(
        -0.85 * np.clip(ranks / rank_scale, 0.0, 12.0)
    )
    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)

    candidate_support = 0.58 + 0.42 * mutual_rank

    alternative_scale = np.sqrt(
        alternatives[:, None] * alternatives[None, :]
    )
    margin = (
        alternative_scale - d
    ) / (
        alternative_scale + d + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_bonus = 1.0 + 0.07 * margin

    local_affinity = np.exp(
        -np.clip(d / (local_scale[:, None] + eps), 0.0, 30.0)
    )
    np.fill_diagonal(local_affinity, 0.0)

    neighborhood_overlap = local_affinity @ local_affinity.T
    neighborhood_norm = np.sqrt(
        np.sum(local_affinity ** 2, axis=1)[:, None]
        * np.sum(local_affinity ** 2, axis=1)[None, :]
    )
    compatibility = np.divide(
        neighborhood_overlap,
        neighborhood_norm + eps,
        out=np.zeros_like(neighborhood_overlap),
        where=neighborhood_norm > 0.0,
    )
    compatibility = np.clip(compatibility, 0.0, 1.0)
    compatibility_factor = 0.96 + 0.04 * compatibility

    score = (
        distance_affinity
        * endpoint_affinity
        * mutual_rank
        * candidate_support
        * margin_bonus
        * compatibility_factor
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
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
