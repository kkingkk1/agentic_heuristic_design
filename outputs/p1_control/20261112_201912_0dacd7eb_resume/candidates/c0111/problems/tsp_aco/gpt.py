import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    finite_positive &= ~np.eye(n, dtype=bool)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[finite_positive]
    fallback = float(np.max(positive_values))

    work = np.where(finite_positive, distances, fallback)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work) & (work > 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_work = work[valid]
    fallback = float(np.max(positive_work))
    safe = np.where(valid, work, fallback)
    np.fill_diagonal(safe, np.inf)

    eps = np.finfo(float).eps

    # Local scale estimated from the closest sqrt(n)-style neighborhood.
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest_values = np.partition(safe, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = safe / (pair_scale + eps)

    # Distance affinity: shorter-than-local-scale edges receive higher scores.
    affinity = np.exp(-np.minimum(relative_length, 50.0))
    affinity[~valid] = 0.0

    # Bidirectional endpoint preference based on local edge ranks.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    positions = np.arange(n)[None, :]
    ranks[np.arange(n)[:, None], order] = positions
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~valid] = 0.0
    bidirectional_rank = np.sqrt(rank_score * rank_score.T)

    # Soft mutual-neighborhood support.
    neighborhood = ranks < k
    mutual_support = np.sqrt(
        neighborhood.astype(float) * neighborhood.T.astype(float)
    )
    mutual_support[~valid] = 0.0

    # Edge indispensability: compare the direct edge with its best two-hop route.
    two_hop = safe[:, None, :] + safe[None, :, :]
    intermediate_ok = np.isfinite(safe)
    two_hop_ok = (
        intermediate_ok[:, None, :]
        & intermediate_ok[None, :, :]
    )
    two_hop_ok &= ~np.eye(n, dtype=bool)[:, :, None]
    two_hop_ok &= ~np.eye(n, dtype=bool)[None, :, :]
    two_hop[~two_hop_ok] = np.inf

    best_alternative = np.min(two_hop, axis=2)
    has_alternative = np.isfinite(best_alternative) & valid

    relative_gap = np.zeros((n, n), dtype=float)
    relative_gap[has_alternative] = (
        best_alternative[has_alternative] - work[has_alternative]
    ) / (pair_scale[has_alternative] + eps)

    # Positive gap means the direct edge is cheaper than every two-hop substitute.
    indispensability = np.clip(relative_gap, 0.0, 50.0)
    indispensability = 1.0 - np.exp(-indispensability)
    indispensability[~valid] = 0.0

    score = (
        0.48 * affinity
        + 0.24 * bidirectional_rank
        + 0.14 * mutual_support
        + 0.14 * indispensability
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
