import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Construct a distance-aware stochastic-sampling prior for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    # Symmetrize valid distances while preserving one-sided valid entries.
    finite_d = np.where(valid, d, 0.0)
    valid_sum = valid.astype(float) + valid.T.astype(float)
    symmetric_d = np.divide(
        finite_d + finite_d.T,
        valid_sum,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid_sum > 0,
    )
    np.fill_diagonal(symmetric_d, np.inf)

    positive = symmetric_d[valid & (symmetric_d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    finite_edges = symmetric_d[valid]
    fallback = float(np.max(finite_edges)) if finite_edges.size else scale
    fallback = max(fallback, scale)

    safe_d = np.where(valid, symmetric_d, fallback)

    # Global shortness and relative shortness to both endpoint neighborhoods.
    shortness = np.exp(-safe_d / scale)

    k = min(3, n - 1)
    local_values = np.where(valid, symmetric_d, np.inf)
    local_scale = np.partition(local_values, kth=k - 1, axis=1)[:, k - 1]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_shortness = np.exp(
        -safe_d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    )

    # Reciprocal neighborhood rank preference.
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, np.sqrt(n))
    rank_preference = np.exp(-ranks / rank_scale)
    rank_preference = 0.5 * (
        rank_preference + rank_preference.T
    )

    nearest = order[:, 0]
    mutual_support = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if valid[i, j] and nearest[j] == i:
            mutual_support[i, j] = 1.0
            mutual_support[j, i] = 1.0

    score = (
        0.50 * shortness
        + 0.30 * relative_shortness
        + 0.15 * rank_preference
        + 0.05 * mutual_support
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
