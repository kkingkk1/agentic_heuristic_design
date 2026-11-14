import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-adaptive edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= off_diagonal

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    d = 0.5 * (repaired + repaired.T)

    symmetric_valid = valid & valid.T

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive_scale = float(np.median(positive))
    scale_floor = max(positive_scale * 1e-10, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(
        work,
        kth=k - 1,
        axis=1,
    )[:, :k]

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        scale_floor,
    )

    if n > 2:
        alternative = np.partition(work, kth=1, axis=1)[:, 1]
    else:
        alternative = nearest[:, 0]

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
        -np.clip(relative_distance ** 0.94, 0.0, 40.0)
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
    endpoint_bottleneck = np.minimum(endpoint_i, endpoint_j)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    mutual_rank = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )

    candidate_support = np.sqrt(
        np.exp(-ranks / rank_scale)
        * np.exp(-ranks.T / rank_scale)
    )
    candidate_gate = 0.68 + 0.32 * candidate_support

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = (
        alternative_scale - work
    ) / (
        alternative_scale + work + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    margin_bonus = 1.0 + 0.06 * margin * np.sqrt(
        np.clip(mutual_rank, 0.0, 1.0)
    )

    density_ratio = np.sqrt(
        local_scale[:, None]
        / np.maximum(local_scale[None, :], scale_floor)
    )
    density_balance = np.exp(
        -0.10 * np.abs(
            np.log(np.maximum(density_ratio, eps))
        )
    )

    global_scale = max(float(np.median(positive)), scale_floor)
    global_guard = 0.94 + 0.06 * np.exp(
        -np.clip((work / global_scale) ** 0.9, 0.0, 40.0)
    )

    score = (
        distance_affinity
        * endpoint_compatibility
        * endpoint_bottleneck
        * mutual_rank
        * candidate_gate
        * margin_bonus
        * density_balance
        * global_guard
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
