import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & off_diagonal
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return result

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    work = np.where(valid, distances, fallback)
    work = 0.5 * (work + work.T)
    work = np.maximum(work, 0.0)
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    floor = max(float(np.median(positive)) * 1e-10, eps)

    candidate_count = max(
        1,
        min(n - 1, int(np.ceil(np.sqrt(n)))),
    )
    nearest = np.partition(
        work,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        floor,
    )

    nearest_two = np.partition(work, kth=min(1, n - 2), axis=1)
    best_alternative = nearest_two[:, min(1, n - 2)]
    best_alternative = np.where(
        np.isfinite(best_alternative),
        best_alternative,
        local_scale,
    )
    best_alternative = np.maximum(best_alternative, floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = work / (pair_scale + eps)

    affinity = 1.0 / (1.0 + relative_distance)
    affinity *= affinity

    endpoint_i = 1.0 / (
        1.0 + work / (local_scale[:, None] + eps)
    )
    endpoint_j = 1.0 / (
        1.0 + work / (local_scale[None, :] + eps)
    )
    compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(work, axis=1, kind="mergesort")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n)[None, :]

    rank_weight = float(max(candidate_count, 1))
    mutual_rank = np.exp(
        -(rank + rank.T) / (2.0 * rank_weight)
    )

    alternative_scale = np.sqrt(
        best_alternative[:, None] * best_alternative[None, :]
    )
    margin = (
        alternative_scale - work
    ) / (
        alternative_scale + work + floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    score = (
        affinity
        * compatibility
        * mutual_rank
        * (1.0 + 0.12 * margin)
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
