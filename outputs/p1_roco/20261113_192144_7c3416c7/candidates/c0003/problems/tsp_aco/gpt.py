import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    if finite_positive.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_positive) * 10.0 + np.mean(finite_positive))
    x = np.where(np.isfinite(d) & (d >= 0), d, fallback)
    np.fill_diagonal(x, 0.0)

    positive = x[(x > 0) & np.isfinite(x)]
    if positive.size == 0:
        return np.zeros_like(x, dtype=float)

    base_scale = float(np.median(positive))
    if not np.isfinite(base_scale) or base_scale <= 0:
        base_scale = 1.0

    row_sorted = np.sort(np.where(np.eye(n, dtype=bool), np.inf, x), axis=1)
    nearest = row_sorted[:, 0]
    nearest = np.where(np.isfinite(nearest) & (nearest > 0), nearest, base_scale)

    scales = np.array(
        [
            base_scale,
            float(np.median(nearest)),
            float(np.quantile(positive, 0.25)),
            float(np.quantile(positive, 0.75)),
        ],
        dtype=float,
    )
    scales = np.maximum(scales, base_scale * 1e-6)

    closeness = np.zeros((n, n), dtype=float)
    for scale in scales:
        closeness += np.exp(-x / scale)
    closeness /= float(len(scales))

    ranks = np.empty((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(np.where(np.arange(n) == i, np.inf, x[i]), kind="mergesort")
        ranks[i, order] = np.arange(n, dtype=float)

    rank_strength = np.exp(-ranks / max(1.0, np.sqrt(n)))
    np.fill_diagonal(rank_strength, 0.0)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    neighborhood = ranks < float(k)
    np.fill_diagonal(neighborhood, False)

    mutual_support = neighborhood.astype(float) * neighborhood.T.astype(float)
    one_sided_support = neighborhood.astype(float) + neighborhood.T.astype(float)

    competition = np.zeros((n, n), dtype=float)
    for i in range(n):
        local_scale = max(nearest[i], base_scale * 1e-6)
        competition[i] = np.exp(-x[i] / local_scale)
    np.fill_diagonal(competition, 0.0)

    edge_competition = np.sqrt(competition * competition.T)
    exclusivity = 1.0 / (1.0 + 2.0 * edge_competition)

    score = (
        (0.45 * closeness + 0.35 * rank_strength + 0.20 * mutual_support)
        * (1.0 + 0.35 * one_sided_support)
        * exclusivity
    )

    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    score /= row_max

    return score.astype(float, copy=False)
