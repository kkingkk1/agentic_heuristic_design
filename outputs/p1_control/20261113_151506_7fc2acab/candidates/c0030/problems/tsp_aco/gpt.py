import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic-sampling priors for directed TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    positive_finite = finite & (distances > 0)

    fallback = np.max(distances[positive_finite]) if np.any(positive_finite) else 1.0
    fallback = max(float(fallback), 1.0)

    d = np.where(finite & (distances >= 0), distances, fallback)
    np.fill_diagonal(d, fallback)

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    eps = np.finfo(float).eps

    # Signal 1: direct short-edge preference.
    inv_distance = 1.0 / (d + eps)

    # Signal 2: reciprocal local rank preference.
    ranks = np.empty_like(d)
    for i in range(n):
        order = np.argsort(d[i], kind="mergesort")
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)
        ranks[i] = rank
    rank_signal = 1.0 / (1.0 + ranks)

    # Signal 3: compatibility with the destination's own neighborhood.
    # An edge is favored when its destination has several short continuations.
    neighborhood_scale = np.partition(
        np.where(valid, d, np.inf), kth=min(2, n - 1), axis=1
    )[:, min(2, n - 1)]
    neighborhood_scale = np.maximum(neighborhood_scale, eps)
    continuation_signal = 1.0 / (d + neighborhood_scale[:, None] + eps)

    def row_normalize(x: np.ndarray) -> np.ndarray:
        x = np.where(valid & np.isfinite(x), x, 0.0)
        row_max = np.max(x, axis=1, keepdims=True)
        row_max = np.maximum(row_max, eps)
        return x / row_max

    a = row_normalize(inv_distance)
    b = row_normalize(rank_signal)
    c = row_normalize(continuation_signal)

    score = np.power(a + eps, 0.55)
    score *= np.power(b + eps, 0.25)
    score *= np.power(c + eps, 0.20)
    score = np.where(valid, score, 0.0)

    # Remove numerical scale differences while retaining positive sampling mass.
    row_sum = score.sum(axis=1, keepdims=True)
    score = np.divide(score, row_sum, out=np.zeros_like(score), where=row_sum > 0)

    for i in range(n):
        if not np.any(score[i] > 0):
            candidates = np.flatnonzero(valid[i])
            if candidates.size:
                score[i, candidates[np.argmin(d[i, candidates])]] = 1.0

    return score
