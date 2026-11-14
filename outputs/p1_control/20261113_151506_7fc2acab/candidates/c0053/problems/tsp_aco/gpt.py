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

    finite_nonnegative = np.isfinite(distances) & (distances >= 0)
    positive_finite = np.isfinite(distances) & (distances > 0)

    if np.any(positive_finite):
        fallback = float(np.max(distances[positive_finite]))
        fallback = max(fallback, 1.0)
    else:
        fallback = 1.0

    d = np.where(finite_nonnegative, distances, fallback).astype(float)
    np.fill_diagonal(d, fallback)

    valid = ~np.eye(n, dtype=bool)
    eps = np.finfo(float).eps

    # Direct edge attractiveness.
    inverse_distance = 1.0 / np.maximum(d, eps)

    # Reciprocal rank among outgoing edges.
    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_signal = 1.0 / (1.0 + ranks)

    # Destination neighborhood potential:
    # short outgoing alternatives from j make i -> j more compatible.
    masked_d = np.where(valid, d, np.inf)
    neighborhood_width = min(3, n - 1)
    nearest = np.partition(
        masked_d, kth=neighborhood_width - 1, axis=1
    )[:, :neighborhood_width]
    neighborhood_mean = np.mean(nearest, axis=1)
    neighborhood_mean = np.maximum(neighborhood_mean, eps)

    # Two-edge lookahead: i -> j followed by a good continuation from j.
    lookahead = 1.0 / (
        d + neighborhood_mean[None, :] + 0.5 * neighborhood_mean[:, None] + eps
    )

    def normalize_rows(signal: np.ndarray) -> np.ndarray:
        signal = np.where(valid & np.isfinite(signal), signal, 0.0)
        maxima = np.max(signal, axis=1, keepdims=True)
        return np.divide(
            signal,
            np.maximum(maxima, eps),
            out=np.zeros_like(signal),
            where=maxima > 0,
        )

    direct = normalize_rows(inverse_distance)
    ranked = normalize_rows(rank_signal)
    compatible = normalize_rows(lookahead)

    score = (
        np.power(direct + eps, 0.60)
        * np.power(ranked + eps, 0.20)
        * np.power(compatible + eps, 0.20)
    )
    score = np.where(valid & np.isfinite(score), score, 0.0)

    row_sums = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_sums,
        out=np.zeros_like(score),
        where=row_sums > 0,
    )

    missing = np.flatnonzero(np.sum(score > 0, axis=1) == 0)
    for i in missing:
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            j = candidates[np.argmin(d[i, candidates])]
            score[i, j] = 1.0

    return score
