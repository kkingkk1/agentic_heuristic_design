import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

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

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    d = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(d, np.inf)

    positive_scale = float(np.median(positive))
    scale_floor = max(positive_scale * 1e-10, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    local_neighbors = np.partition(
        d,
        kth=k - 1,
        axis=1,
    )[:, :k]

    local_mean = np.maximum(
        np.mean(local_neighbors, axis=1),
        scale_floor,
    )
    local_std = np.std(local_neighbors, axis=1)
    local_dispersion = local_std / local_mean

    local_scale = local_mean * (
        0.92 + 0.18 * local_dispersion / (1.0 + local_dispersion)
    )
    local_scale = np.maximum(local_scale, scale_floor)

    if n > 2:
        alternative = np.partition(d, kth=1, axis=1)[:, 1]
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

    normalized_distance = d / (pair_scale + scale_floor)
    distance_affinity = np.exp(
        -np.clip(normalized_distance ** 0.94, 0.0, 50.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            d / (local_scale[:, None] + scale_floor),
            0.0,
            50.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            d / (local_scale[None, :] + scale_floor),
            0.0,
            50.0,
        )
    )

    endpoint_affinity = np.sqrt(endpoint_i * endpoint_j)
    bottleneck_affinity = np.minimum(endpoint_i, endpoint_j)

    work = d.copy()
    row_order = np.argsort(work, axis=1, kind="mergesort")
    row_rank = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    row_rank[rows, row_order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / np.sqrt(1.0 + row_rank)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    adaptive_width = k * (
        0.88 + 0.34 * local_dispersion / (1.0 + local_dispersion)
    )
    adaptive_width = np.maximum(adaptive_width, 1.0)

    support_i = np.exp(
        -np.clip(row_rank / adaptive_width[:, None], 0.0, 50.0)
    )
    support_j = np.exp(
        -np.clip(row_rank.T / adaptive_width[None, :], 0.0, 50.0)
    )

    mutual_support = np.sqrt(support_i * support_j)
    candidate_gate = 0.72 + 0.28 * mutual_support

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin_bonus = 1.0 + 0.08 * np.clip(margin, 0.0, 1.0)

    density_ratio = np.sqrt(
        local_scale[:, None] /
        np.maximum(local_scale[None, :], scale_floor)
    )
    density_balance = np.exp(
        -0.14 * np.abs(np.log(np.maximum(density_ratio, eps)))
    )

    score = (
        distance_affinity
        * endpoint_affinity
        * bottleneck_affinity
        * mutual_rank
        * candidate_gate
        * margin_bonus
        * density_balance
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
