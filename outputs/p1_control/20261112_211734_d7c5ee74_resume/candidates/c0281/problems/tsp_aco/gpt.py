import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric local-neighborhood edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    replacement = (
        float(np.max(finite_values))
        if finite_values.size
        else fallback
    )
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = fallback

    d = np.where(np.isfinite(d), d, replacement)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    ordered = np.sort(d, axis=1)
    nearest = ordered[:, :k]

    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    alternative = (
        ordered[:, 1]
        if n > 2
        else ordered[:, 0]
    )
    alternative = np.maximum(alternative, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = d / (pair_scale + eps)

    distance_affinity = np.exp(
        -np.clip(normalized_length ** 0.92, 0.0, 40.0)
    )

    endpoint_ratio_i = d / (local_scale[:, None] + eps)
    endpoint_ratio_j = d / (local_scale[None, :] + eps)

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_ratio_i, 0.0, 40.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_ratio_j, 0.0, 40.0)
    )

    endpoint_compatibility = np.sqrt(
        endpoint_affinity_i * endpoint_affinity_j
    )
    bottleneck_affinity = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    rank_i = np.exp(-np.clip(ranks / rank_scale, 0.0, 40.0))
    rank_j = rank_i.T
    mutual_rank = np.sqrt(rank_i * rank_j)

    candidate_gate = 0.62 + 0.38 * mutual_rank

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = (
        alternative_scale - d
    ) / (
        alternative_scale + d + eps
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_bonus = 1.0 + 0.08 * margin * mutual_rank

    local_affinity = np.exp(
        -np.clip(d / (local_scale[:, None] + eps), 0.0, 40.0)
    )
    np.fill_diagonal(local_affinity, 0.0)

    support = local_affinity @ local_affinity.T
    support_norm = np.sqrt(
        np.sum(local_affinity ** 2, axis=1)[:, None]
        * np.sum(local_affinity ** 2, axis=1)[None, :]
    )
    normalized_support = np.clip(
        support / (support_norm + eps),
        0.0,
        1.0,
    )
    support_factor = 0.97 + 0.03 * normalized_support

    score = (
        distance_affinity
        * endpoint_compatibility
        * bottleneck_affinity
        * mutual_rank
        * candidate_gate
        * margin_bonus
        * support_factor
    )

    score = 0.5 * (score + score.T)
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
