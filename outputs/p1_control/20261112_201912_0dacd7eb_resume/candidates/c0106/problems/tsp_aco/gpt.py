import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    positive_values = distances[finite_positive]
    fallback = float(np.max(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    d = np.asarray(distances, dtype=float).copy()
    d[~np.isfinite(d)] = fallback
    d[d < 0.0] = 0.0
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    work = d.copy()
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / (endpoint_scale + np.finfo(float).eps)

    closeness = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_term = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_term * rank_term.T)

    endpoint_fit_i = np.exp(
        -np.clip(d / local_scale[:, None], 0.0, 50.0)
    )
    endpoint_fit_j = np.exp(
        -np.clip(d / local_scale[None, :], 0.0, 50.0)
    )
    compatibility = np.sqrt(endpoint_fit_i * endpoint_fit_j)

    alternative = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        np.finfo(float).eps,
    )
    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )

    margin = np.maximum(
        substitute_scale - d,
        0.0,
    ) / (
        substitute_scale + d + np.finfo(float).eps
    )
    margin = np.clip(margin, 0.0, 1.0)

    base = (
        closeness
        * (0.55 + 0.45 * mutual_rank)
        * (0.65 + 0.35 * compatibility)
    )
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    local_margin_rank_order = np.argsort(
        -margin,
        axis=1,
        kind="mergesort",
    )
    local_margin_rank = np.empty_like(
        local_margin_rank_order,
        dtype=float,
    )
    local_margin_rank[rows, local_margin_rank_order] = (
        np.arange(n, dtype=float)[None, :]
    )

    candidate_count = max(1, int(np.ceil(0.20 * (n - 1))))
    high_margin = (
        (local_margin_rank < candidate_count)
        & off_diagonal
    )
    mutual_high_margin = high_margin & high_margin.T

    bonus = 1.0 + 0.10 * margin * mutual_high_margin
    score = base * bonus
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
