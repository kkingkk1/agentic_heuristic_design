import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    valid_values = d[finite]
    if valid_values.size == 0:
        return np.ones_like(d, dtype=float) - np.eye(n)

    fallback = np.max(valid_values) if valid_values.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback * 10.0)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    np.fill_diagonal(d, np.inf)

    finite_edges = np.isfinite(d)
    positive = d[finite_edges & (d > 0)]
    scale = np.median(positive) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_d = np.where(np.isfinite(d), d, scale * 100.0)

    inverse = 1.0 / (safe_d + scale * 1e-6)
    inverse /= np.max(inverse, axis=1, keepdims=True) + 1e-12

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.0, n / 4.0))
    rank_score /= np.max(rank_score, axis=1, keepdims=True) + 1e-12

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score = 0.55 * inverse + 0.30 * rank_score + 0.15 * mutual
    score = 0.5 * (score + score.T)

    score[~finite_edges] = 0.0
    np.fill_diagonal(score, 0.0)

    positive_scores = score[score > 0]
    floor = (
        np.min(positive_scores) * 1e-3
        if positive_scores.size
        else 1e-12
    )
    score[finite_edges & (~np.eye(n, dtype=bool))] += floor

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float)
