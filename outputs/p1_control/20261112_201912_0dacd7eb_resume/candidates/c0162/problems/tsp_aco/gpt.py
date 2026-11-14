import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]
    fallback = float(np.max(positive)) if positive.size else 1.0
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
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    positive_values = d[d > 0.0]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
        global_upper = float(np.percentile(positive_values, 75.0))
    else:
        global_scale = fallback
        global_upper = fallback

    global_scale = max(global_scale, eps)
    global_upper = max(global_upper, global_scale)

    k = max(1, min(n - 1, int(np.ceil(1.25 * np.sqrt(n)))))
    ordered = np.partition(work, k - 1, axis=1)[:, :k]
    ordered.sort(axis=1)

    weights = np.linspace(1.0, 0.55, k, dtype=float)
    weights /= np.sum(weights)

    local_mean = np.sum(ordered * weights[None, :], axis=1)
    local_mean = np.maximum(local_mean, eps)

    local_median = np.maximum(
        np.median(ordered, axis=1),
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

    local_closeness = np.exp(
        -np.clip(relative_mean ** 0.92, 0.0, 50.0)
    )
    median_closeness = np.exp(
        -np.clip(relative_median ** 0.85, 0.0, 50.0)
    )

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    normalized_rank = ranks / max(float(n - 1), 1.0)
    rank_score = 1.0 / (1.0 + 2.0 * normalized_rank)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    mutual_neighborhood = (
        (normalized_rank <= 0.30)
        & (normalized_rank.T <= 0.30)
    )
    rank_bonus = np.where(mutual_neighborhood, 1.12, 1.0)

    endpoint_ratio_i = d / (local_mean[:, None] + eps)
    endpoint_ratio_j = d / (local_mean[None, :] + eps)
    mismatch = np.abs(
        np.log1p(endpoint_ratio_i)
        - np.log1p(endpoint_ratio_j)
    )
    endpoint_compatibility = np.exp(
        -np.clip(
            0.55 * (
                np.log1p(endpoint_ratio_i)
                + np.log1p(endpoint_ratio_j)
            )
            + 0.35 * mismatch,
            0.0,
            50.0,
        )
    )

    alternative_i = ordered[:, min(1, k - 1)]
    alternative_j = ordered[:, min(1, k - 1)]
    substitute_scale = np.sqrt(
        alternative_i[:, None] * alternative_j[None, :]
    )
    margin = substitute_scale / (
        d + substitute_scale + eps
    )
    margin = np.clip(margin, 0.0, 1.0)

    global_gate = np.exp(
        -np.clip((d / global_scale) ** 0.80, 0.0, 50.0)
    )
    global_gate = 0.35 + 0.65 * global_gate

    tail_guard = 1.0 / (
        1.0
        + 0.20 * np.maximum(
            d / global_upper - 1.0,
            0.0,
        ) ** 1.25
    )

    score = (
        local_closeness ** 1.15
        * median_closeness ** 0.35
        * mutual_rank ** 1.05
        * rank_bonus
        * endpoint_compatibility ** 1.10
        * global_gate
        * tail_guard
        * (0.94 + 0.10 * margin)
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
