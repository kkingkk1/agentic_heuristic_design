import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite = np.isfinite(distances)
    positive = finite & (distances > 0.0)
    np.fill_diagonal(positive, False)

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive]
    fallback = float(np.percentile(positive_values, 85.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
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

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n))) + 1))
    kth = min(k, n - 1)

    partitioned = np.partition(d, kth, axis=1)
    neighborhood = partitioned[:, :k]

    local_scale = np.maximum(
        np.mean(neighborhood, axis=1),
        eps,
    )

    if k < n - 1:
        alternative = np.maximum(partitioned[:, k], eps)
    else:
        alternative = np.maximum(partitioned[:, k - 1], eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = d / np.maximum(pair_scale, eps)
    distance_affinity = np.exp(
        -np.clip(relative_length ** 0.88, 0.0, 50.0)
    )

    endpoint_ratio_i = d / np.maximum(local_scale[:, None], eps)
    endpoint_ratio_j = d / np.maximum(local_scale[None, :], eps)

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.80, 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.80, 0.0, 50.0)
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
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    normalized_rank = ranks / max(float(n - 1), 1.0)
    reciprocal_rank = np.exp(
        -1.40
        * np.sqrt(
            np.clip(
                normalized_rank * normalized_rank.T,
                0.0,
                1.0,
            )
        )
    )

    candidate_width = float(max(k, 1))
    candidate_support_i = np.exp(
        -np.clip(ranks / candidate_width, 0.0, 50.0)
    )
    candidate_support_j = np.exp(
        -np.clip(ranks.T / candidate_width, 0.0, 50.0)
    )
    candidate_gate = 0.54 + 0.46 * np.sqrt(
        candidate_support_i * candidate_support_j
    )

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + eps
    )
    margin_bonus = 1.0 + 0.10 * np.clip(margin, 0.0, 1.0)

    positive_values_after = d[np.isfinite(d) & (d > 0.0)]
    global_scale = (
        float(np.median(positive_values_after))
        if positive_values_after.size
        else finite_max
    )
    global_scale = max(global_scale, eps)

    global_guard = 0.92 + 0.08 * np.exp(
        -np.clip((d / global_scale) ** 0.88, 0.0, 50.0)
    )

    score = (
        distance_affinity
        * endpoint_compatibility
        * bottleneck_affinity
        * reciprocal_rank
        * candidate_gate
        * margin_bonus
        * global_guard
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
