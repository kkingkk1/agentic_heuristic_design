import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized endpoint-aware edge-priority indicators for stochastic TSP sampling."""
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

    fallback = float(np.percentile(positive, 85.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive_scale = float(np.median(positive))
    scale_floor = max(
        positive_scale * 1e-10,
        np.finfo(float).eps,
    )

    neighborhood_fraction = 0.85
    k = max(
        1,
        min(
            n - 1,
            int(np.ceil(neighborhood_fraction * np.sqrt(n))),
        ),
    )

    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    if n > 2:
        alternative = ordered[:, 1]
    else:
        alternative = ordered[:, 0]

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
        -np.clip(1.10 * normalized_distance, 0.0, 40.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            0.90 * d / (local_scale[:, None] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            0.90 * d / (local_scale[None, :] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(0.85 * k, 1.0))
    raw_rank_support = np.exp(
        -0.35 * (ranks + ranks.T) / rank_scale
    )

    # Keep ranking informative while imposing a nonzero exploration floor.
    candidate_support = 0.65 + 0.35 * raw_rank_support

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_bonus = 1.0 + 0.08 * margin

    score = (
        distance_affinity
        * endpoint_compatibility
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
