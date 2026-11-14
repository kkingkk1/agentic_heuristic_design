import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    finite_positive &= ~diagonal

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[finite_positive]
    fallback = float(np.percentile(positive_values, 90.0))

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive_values))

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
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

    positive_after = d[np.isfinite(d) & (d > 0.0)]
    scale_floor = (
        max(float(np.median(positive_after)) * 1e-10, eps)
        if positive_after.size
        else eps
    )

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    partitioned = np.partition(d, k - 1, axis=1)
    neighborhood = partitioned[:, :k]
    local_scale = np.maximum(neighborhood.mean(axis=1), scale_floor)

    if k < n - 1:
        alternative = partitioned[:, k]
    else:
        alternative = partitioned[:, k - 1]

    alternative = np.maximum(alternative, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / np.maximum(pair_scale, scale_floor)
    distance_affinity = np.exp(
        -np.clip(relative_distance ** 0.95, 0.0, 50.0)
    )

    endpoint_ratio_i = d / np.maximum(local_scale[:, None], scale_floor)
    endpoint_ratio_j = d / np.maximum(local_scale[None, :], scale_floor)

    endpoint_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.9, 0.0, 50.0)
    )
    endpoint_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.9, 0.0, 50.0)
    )

    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)
    bottleneck_compatibility = np.minimum(endpoint_i, endpoint_j)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    normalized_rank = ranks / max(float(n - 1), 1.0)

    mutual_rank = np.sqrt(
        (1.0 / (1.0 + ranks))
        * (1.0 / (1.0 + ranks.T))
    )

    rank_affinity = np.exp(
        -0.35 * np.clip(normalized_rank + normalized_rank.T, 0.0, 2.0)
    )

    candidate_width = float(max(k, 1))
    candidate_support = np.exp(
        -0.5 * (ranks + ranks.T) / candidate_width
    )
    candidate_gate = 0.60 + 0.40 * candidate_support

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (
        d + substitute_scale + scale_floor
    )
    margin_bonus = 1.0 + 0.08 * np.clip(margin, 0.0, 1.0)

    score = (
        distance_affinity
        * mutual_rank
        * rank_affinity
        * candidate_gate
        * (0.78 + 0.22 * endpoint_compatibility)
        * (0.88 + 0.12 * bottleneck_compatibility)
        * margin_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~diagonal
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 83.0))
        score *= np.where(score >= cutoff, 1.03, 1.0)

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
