import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-adaptive edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & off_diagonal
    )

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)

    work = repaired.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    scale_floor = max(float(np.median(positive)) * 1e-10, eps)

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    local_neighbors = np.partition(
        work,
        kth=k - 1,
        axis=1,
    )[:, :k]

    local_scale = np.maximum(
        np.mean(local_neighbors, axis=1),
        scale_floor,
    )

    if n > 2:
        alternative = np.partition(work, kth=1, axis=1)[:, 1]
    else:
        alternative = local_neighbors[:, 0]

    alternative = np.where(
        np.isfinite(alternative),
        alternative,
        local_scale,
    )
    alternative = np.maximum(alternative, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = work / (pair_scale + scale_floor)

    distance_affinity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            work / (local_scale[:, None] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            work / (local_scale[None, :] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    mutual_rank = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = (
        alternative_scale - work
    ) / (
        alternative_scale + work + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    rank_gate = 0.60 + 0.40 * np.sqrt(
        np.clip(mutual_rank, 0.0, 1.0)
    )
    margin_bonus = 1.0 + 0.08 * margin * rank_gate

    score = (
        distance_affinity
        * endpoint_compatibility
        * mutual_rank
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
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
