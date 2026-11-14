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

    fallback = (
        float(np.max(distances[positive_finite]))
        if np.any(positive_finite)
        else 1.0
    )
    fallback = max(fallback, 1.0)

    d = np.where(finite & (distances >= 0), distances, fallback)
    np.fill_diagonal(d, fallback)

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    eps = np.finfo(float).eps

    # Signal 1: direct preference for short edges.
    inv_distance = 1.0 / (d + eps)

    # Signal 2: reciprocal local-rank preference.
    ranks = np.empty_like(d)
    for i in range(n):
        order = np.argsort(d[i], kind="mergesort")
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)
        ranks[i] = rank

    rank_signal = 1.0 / (1.0 + ranks)

    # Signal 3: two-edge compatibility with the destination's neighborhood.
    masked_distances = np.where(valid, d, np.inf)
    neighborhood_scale = np.partition(
        masked_distances,
        kth=min(2, n - 1),
        axis=1,
    )[:, min(2, n - 1)]
    neighborhood_scale = np.maximum(neighborhood_scale, eps)

    continuation_signal = 1.0 / (
        d + neighborhood_scale[None, :] + eps
    )

    def row_normalize(signal: np.ndarray) -> np.ndarray:
        signal = np.where(valid & np.isfinite(signal), signal, 0.0)
        row_max = np.max(signal, axis=1, keepdims=True)
        row_max = np.maximum(row_max, eps)
        return signal / row_max

    direct = row_normalize(inv_distance)
    ranked = row_normalize(rank_signal)
    compatible = row_normalize(continuation_signal)

    # Weighted geometric fusion: direct distance remains dominant.
    score = np.power(direct + eps, 0.60)
    score *= np.power(ranked + eps, 0.22)
    score *= np.power(compatible + eps, 0.18)
    score = np.where(valid, score, 0.0)

    row_sum = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_sum,
        out=np.zeros_like(score),
        where=row_sum > 0,
    )

    for i in range(n):
        if not np.any(score[i] > 0):
            candidates = np.flatnonzero(valid[i])
            if candidates.size:
                best = candidates[np.argmin(d[i, candidates])]
                score[i, best] = 1.0

    return score
