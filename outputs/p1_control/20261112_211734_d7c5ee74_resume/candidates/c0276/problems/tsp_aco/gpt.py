import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

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
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[np.isfinite(d)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    finite_max = float(np.max(finite_offdiag))
    if not np.isfinite(finite_max) or finite_max <= 0.0:
        finite_max = fallback

    d = np.where(np.isfinite(d), d, finite_max)
    np.fill_diagonal(d, np.inf)

    scale_floor = max(
        float(np.median(positive)) * 1e-10,
        eps,
    )

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    partitioned = np.partition(d, k - 1, axis=1)
    neighborhood = partitioned[:, :k]

    local_scale = np.maximum(
        np.mean(neighborhood, axis=1),
        scale_floor,
    )

    if k < n - 1:
        alternative = partitioned[:, k]
    else:
        alternative = partitioned[:, k - 1]

    alternative = np.maximum(alternative, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = d / np.maximum(pair_scale, scale_floor)
    distance_affinity = np.exp(
        -np.clip(normalized_distance ** 0.9, 0.0, 50.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            d / np.maximum(local_scale[:, None], scale_floor),
            0.0,
            50.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            d / np.maximum(local_scale[None, :], scale_floor),
            0.0,
            50.0,
        )
    )

    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)
    bottleneck_compatibility = np.minimum(endpoint_i, endpoint_j)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))

    mutual_rank = np.sqrt(
        (1.0 / (1.0 + ranks))
        * (1.0 / (1.0 + ranks.T))
    )

    candidate_support = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )
    candidate_gate = 0.60 + 0.40 * candidate_support

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin_bonus = 1.0 + 0.07 * np.clip(margin, 0.0, 1.0)

    score = (
        distance_affinity
        * endpoint_compatibility
        * bottleneck_compatibility
        * mutual_rank
        * candidate_gate
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
