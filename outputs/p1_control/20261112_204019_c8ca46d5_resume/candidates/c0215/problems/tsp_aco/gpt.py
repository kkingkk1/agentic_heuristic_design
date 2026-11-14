import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    positive = distances[finite_positive]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
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
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    neighborhood = np.partition(work, k - 1, axis=1)[:, :k]
    neighborhood.sort(axis=1)

    weights = np.linspace(1.0, 0.60, k, dtype=float)
    weights /= weights.sum()

    local_mean = np.maximum(
        np.sum(neighborhood * weights[None, :], axis=1),
        eps,
    )
    local_median = np.maximum(
        np.median(neighborhood, axis=1),
        eps,
    )

    pair_mean = np.sqrt(local_mean[:, None] * local_mean[None, :])
    pair_median = np.sqrt(
        local_median[:, None] * local_median[None, :]
    )

    relative_mean = d / (pair_mean + eps)
    relative_median = d / (pair_median + eps)

    distance_affinity = np.exp(
        -np.clip(relative_mean ** 0.92, 0.0, 50.0)
    )
    median_affinity = np.exp(
        -np.clip(relative_median ** 0.86, 0.0, 50.0)
    )

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    normalized_rank = ranks / max(float(n - 1), 1.0)
    rank_score = 1.0 / (1.0 + 2.0 * normalized_rank)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    candidate_gate = 0.88 + 0.12 * np.exp(
        -0.5 * (normalized_rank + normalized_rank.T)
    )

    endpoint_i = 1.0 / (1.0 + d / (local_mean[:, None] + eps))
    endpoint_j = 1.0 / (1.0 + d / (local_mean[None, :] + eps))
    mutual_competition = np.sqrt(endpoint_i * endpoint_j)

    alternative = neighborhood[:, min(1, k - 1)]
    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (
        d + substitute_scale + eps
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_factor = 0.96 + 0.06 * margin

    score = (
        distance_affinity
        * median_affinity ** 0.35
        * mutual_rank ** 1.08
        * (0.82 + 0.18 * mutual_competition)
        * candidate_gate
        * margin_factor
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
