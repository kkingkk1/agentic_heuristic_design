import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized edge priorities for stochastic TSP sampling."""
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

    mutually_valid = valid & valid.T

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)

    work = repaired.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    scale_floor = max(
        float(np.median(positive)) * 1e-10,
        eps,
    )

    k = max(1, min(n - 1, int(np.ceil(1.1 * np.sqrt(n)))))

    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(
        0.75 * np.mean(nearest, axis=1)
        + 0.25 * np.median(nearest, axis=1),
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
        -np.clip(relative_distance ** 0.95, 0.0, 40.0)
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
    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    rank_pressure = 0.5 * (
        ranks / rank_scale
        + ranks.T / rank_scale
    )
    candidate_gate = 0.82 + 0.18 * np.exp(
        -np.clip(rank_pressure, 0.0, 8.0)
    )

    competition = np.sqrt(
        1.0 / (1.0 + relative_distance)
        * 1.0 / (1.0 + relative_distance.T)
    )
    competition_factor = 0.88 + 0.12 * competition

    margin_i = (
        alternative[:, None] - work
    ) / (
        alternative[:, None] + work + scale_floor
    )
    margin_j = (
        alternative[None, :] - work
    ) / (
        alternative[None, :] + work + scale_floor
    )
    margin = 0.5 * (
        np.clip(margin_i, 0.0, 1.0)
        + np.clip(margin_j, 0.0, 1.0)
    )
    margin_factor = 1.0 + 0.06 * margin

    local_affinity = np.exp(
        -np.clip(work / (local_scale[:, None] + scale_floor), 0.0, 12.0)
    )
    np.fill_diagonal(local_affinity, 0.0)

    support = local_affinity @ local_affinity.T
    support_norm = np.sqrt(
        np.maximum(
            np.sum(local_affinity * local_affinity, axis=1)[:, None]
            * np.sum(local_affinity * local_affinity, axis=1)[None, :],
            eps,
        )
    )
    common_neighborhood = np.clip(
        support / (support_norm + eps),
        0.0,
        1.0,
    )
    compatibility_factor = 0.97 + 0.03 * common_neighborhood

    score = (
        distance_affinity
        * endpoint_compatibility
        * reciprocal_rank
        * candidate_gate
        * competition_factor
        * margin_factor
        * compatibility_factor
    )

    score = 0.5 * (score + score.T)
    score[~mutually_valid] = 0.0
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
