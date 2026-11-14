import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    finite_values = d[finite]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    d = np.where(finite, d, fallback)

    eps = np.finfo(float).eps
    k = min(n - 1, max(2, int(np.sqrt(n)) + 1))

    nearest = np.partition(d, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    relative = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_term = 1.0 / (1.0 + relative)

    endpoint_ratio_a = d / local_scale[:, None]
    endpoint_ratio_b = d / local_scale[None, :]
    endpoint_affinity_a = 1.0 / (1.0 + np.sqrt(endpoint_ratio_a))
    endpoint_affinity_b = 1.0 / (1.0 + np.sqrt(endpoint_ratio_b))
    endpoint_term = (
        0.35
        + 0.65 * np.minimum(endpoint_affinity_a, endpoint_affinity_b)
    )

    rank_limit = min(n - 1, max(4, 2 * k))
    candidate_indices = np.argpartition(
        d,
        rank_limit - 1,
        axis=1,
    )[:, :rank_limit]

    candidate_distances = np.take_along_axis(
        d,
        candidate_indices,
        axis=1,
    )
    candidate_order = np.argsort(
        candidate_distances,
        axis=1,
        kind="stable",
    )
    ranked_indices = np.take_along_axis(
        candidate_indices,
        candidate_order,
        axis=1,
    )

    rank_weights = 1.0 - (
        np.arange(rank_limit, dtype=float) / float(rank_limit)
    )
    rank_mask = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    rank_mask[rows, ranked_indices] = rank_weights[None, :]

    reverse_rank = rank_mask.T
    rank_term = 0.5 * (rank_mask + reverse_rank)
    mutual_term = 1.0 + 0.12 * (
        (rank_mask > 0.0) & (reverse_rank > 0.0)
    ).astype(float)

    score = (
        local_term
        * endpoint_term
        * (0.82 + 0.18 * rank_term)
        * mutual_term
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
