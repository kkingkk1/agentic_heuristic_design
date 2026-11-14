import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust endpoint-aware edge priorities for stochastic TSP sampling."""
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
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    d = np.where(np.isfinite(d), d, fallback)

    work = d.copy()
    np.fill_diagonal(work, np.inf)
    np.fill_diagonal(d, 0.0)

    finite_offdiag = work[np.isfinite(work)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_offdiag = finite_offdiag[finite_offdiag > 0.0]
    scale_values = (
        positive_offdiag
        if positive_offdiag.size
        else finite_offdiag
    )

    global_scale = max(
        float(np.median(scale_values)),
        eps,
    )
    global_upper = max(
        float(np.percentile(scale_values, 75.0)),
        global_scale,
    )

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    order = np.argsort(work, axis=1, kind="mergesort")
    neighbors = np.take_along_axis(
        work,
        order[:, :k],
        axis=1,
    )

    weights = np.linspace(1.0, 0.55, k, dtype=float)
    weights /= np.sum(weights)

    local_mean = np.sum(
        neighbors * weights[None, :],
        axis=1,
    )
    local_mean = np.maximum(local_mean, eps)

    local_median = np.maximum(
        np.median(neighbors, axis=1),
        eps,
    )

    pair_mean = np.sqrt(
        local_mean[:, None] * local_mean[None, :]
    )
    pair_median = np.sqrt(
        local_median[:, None] * local_median[None, :]
    )

    relative_mean = d / (pair_mean + eps)
    relative_median = d / (pair_median + eps)

    mean_affinity = np.exp(
        -np.clip(relative_mean ** 0.90, 0.0, 50.0)
    )
    median_affinity = np.exp(
        -np.clip(relative_median ** 0.82, 0.0, 50.0)
    )

    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(float(n - 1), 1.0)
    normalized_rank = ranks / rank_scale

    rank_affinity = np.exp(
        -1.25
        * 0.5
        * (normalized_rank + normalized_rank.T)
    )

    candidate_i = np.exp(
        -np.clip(ranks / max(float(k), 1.0), 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(ranks.T / max(float(k), 1.0), 0.0, 50.0)
    )
    mutual_candidate = np.sqrt(candidate_i * candidate_j)
    candidate_gate = 0.58 + 0.42 * mutual_candidate

    endpoint_ratio_i = d / (local_mean[:, None] + eps)
    endpoint_ratio_j = d / (local_mean[None, :] + eps)

    log_ratio_i = np.log1p(np.maximum(endpoint_ratio_i, 0.0))
    log_ratio_j = np.log1p(np.maximum(endpoint_ratio_j, 0.0))

    endpoint_mean = 0.5 * (log_ratio_i + log_ratio_j)
    endpoint_mismatch = np.abs(log_ratio_i - log_ratio_j)

    endpoint_compatibility = np.exp(
        -np.clip(
            0.62 * endpoint_mean
            + 0.28 * endpoint_mismatch,
            0.0,
            50.0,
        )
    )

    alternative = neighbors[:, min(1, k - 1)]
    alternative_scale = np.sqrt(
        np.maximum(alternative[:, None], eps)
        * np.maximum(alternative[None, :], eps)
    )

    margin = alternative_scale / (
        d + alternative_scale + eps
    )
    margin = np.clip(margin, 0.0, 1.0)

    margin_bonus = 0.94 + 0.12 * margin

    mutual_local = (
        (ranks <= float(k - 1))
        & (ranks.T <= float(k - 1))
    )
    reciprocal_bonus = np.where(
        mutual_local,
        1.10,
        1.0,
    )

    global_affinity = np.exp(
        -np.clip((d / global_scale) ** 0.78, 0.0, 50.0)
    )
    global_gate = 0.40 + 0.60 * global_affinity

    tail_guard = 1.0 / (
        1.0
        + 0.16
        * np.maximum(d / global_upper - 1.0, 0.0) ** 1.20
    )

    score = (
        mean_affinity ** 1.15
        * median_affinity ** 0.35
        * rank_affinity
        * candidate_gate
        * endpoint_compatibility ** 1.10
        * margin_bonus
        * reciprocal_bonus
        * global_gate
        * tail_guard
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
