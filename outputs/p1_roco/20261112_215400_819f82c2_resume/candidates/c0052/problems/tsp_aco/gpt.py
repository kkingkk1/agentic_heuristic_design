import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    d = distance_matrix.copy()
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    safe_d = np.where(finite, d, np.nan)

    nearest = np.nanmin(safe_d, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    low_quantile = np.nanpercentile(safe_d, 35.0, axis=1)
    low_quantile = np.where(np.isfinite(low_quantile), low_quantile, fallback)

    local_scale = 0.6 * nearest + 0.4 * low_quantile
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_preference = np.exp(
        -d / local_scale[:, None],
        where=finite,
        out=np.zeros_like(d, dtype=float),
    )

    row_sums = endpoint_preference.sum(axis=1, keepdims=True)
    endpoint_preference = np.divide(
        endpoint_preference,
        row_sums,
        out=np.zeros_like(endpoint_preference),
        where=row_sums > 0,
    )

    geometric_preference = np.sqrt(
        endpoint_preference * endpoint_preference.T
    )

    nearest_index = np.argmin(d, axis=1)
    nearest_bonus = np.zeros((n, n), dtype=float)
    nearest_bonus[np.arange(n), nearest_index] = 1.0
    mutual_nearest = nearest_bonus * nearest_bonus.T

    ranks = np.argsort(np.argsort(np.where(finite, d, np.inf), axis=1), axis=1)
    rank_scale = max(n - 1, 1)
    reciprocal_rank = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )
    reciprocal_rank[~(finite & finite.T)] = 0.0

    scores = geometric_preference
    scores *= 1.0 + 0.75 * mutual_nearest
    scores *= 0.75 + 0.25 * reciprocal_rank

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores
