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

    scale_floor = max(
        float(np.median(positive)) * 1e-10,
        np.finfo(float).eps,
    )

    fallback = float(np.percentile(positive, 80.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    working = np.where(valid, distances, fallback)
    working = np.maximum(working, 0.0)
    working = 0.5 * (working + working.T)
    np.fill_diagonal(working, np.inf)

    neighborhood_size = max(
        1,
        min(
            n - 1,
            int(np.ceil(0.65 * np.sqrt(n))),
        ),
    )

    ordered = np.sort(working, axis=1)
    local_scale = np.mean(
        ordered[:, :neighborhood_size],
        axis=1,
    )
    local_scale = np.maximum(
        np.nan_to_num(
            local_scale,
            nan=fallback,
            posinf=fallback,
            neginf=scale_floor,
        ),
        scale_floor,
    )

    if n > 2:
        alternative = ordered[:, 1]
    else:
        alternative = ordered[:, 0]

    alternative = np.maximum(
        np.nan_to_num(
            alternative,
            nan=fallback,
            posinf=fallback,
            neginf=scale_floor,
        ),
        scale_floor,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = working / (
        pair_scale + scale_floor
    )

    distance_affinity = np.exp(
        -np.clip(
            1.22 * normalized_distance,
            0.0,
            40.0,
        )
    )

    endpoint_i = np.exp(
        -np.clip(
            0.82 * working
            / (local_scale[:, None] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            0.82 * working
            / (local_scale[None, :] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_compatibility = np.sqrt(
        endpoint_i * endpoint_j
    )

    order = np.argsort(
        working,
        axis=1,
        kind="mergesort",
    )
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(
        0.70 * neighborhood_size,
        1.0,
    )
    candidate_support = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )

    mutual_rank = np.minimum(
        ranks,
        ranks.T,
    )
    mutual_support = np.exp(
        -0.35 * mutual_rank / rank_scale
    )

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        working + alternative_scale + scale_floor
    )
    margin = np.clip(
        margin,
        0.0,
        1.0,
    )
    margin_bonus = 1.0 + 0.05 * margin

    score = (
        distance_affinity
        * endpoint_compatibility
        * candidate_support
        * mutual_support
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
