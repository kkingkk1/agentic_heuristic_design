import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric soft edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    offdiag = ~np.eye(n, dtype=bool)
    valid = finite_nonnegative & finite_nonnegative.T & offdiag

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)
    positive = symmetric[valid & (symmetric > 0.0)]

    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    safe = np.where(valid, np.maximum(symmetric, 0.0), np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + eps)

    inverse_max = float(np.max(inverse))
    inverse_scaled = inverse / max(inverse_max, eps)

    row_sum = np.sum(inverse, axis=1, keepdims=True)
    col_sum = np.sum(inverse, axis=0, keepdims=True)

    row_competition = inverse / np.maximum(row_sum, eps)
    col_competition = inverse / np.maximum(col_sum, eps)
    mutual_competition = np.sqrt(
        np.maximum(row_competition * col_competition, 0.0)
    )

    row_peak = np.max(inverse, axis=1, keepdims=True)
    col_peak = np.max(inverse, axis=0, keepdims=True)

    row_strength = inverse / np.maximum(row_peak, eps)
    col_strength = inverse / np.maximum(col_peak, eps)
    local_exclusivity = np.sqrt(
        np.maximum(row_strength * col_strength, 0.0)
    )

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_temperature = max(1.0, np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_temperature)
    rank_score[~valid] = 0.0

    neighborhood_size = min(n - 1, max(2, int(np.sqrt(n)) + 1))
    top_inverse = np.where(valid, inverse, 0.0)
    if neighborhood_size < n:
        top_inverse = np.partition(
            top_inverse,
            n - neighborhood_size,
            axis=1,
        )[:, n - neighborhood_size:]

    node_potential = np.mean(top_inverse, axis=1)
    node_potential /= max(float(np.max(node_potential)), eps)

    endpoint_potential = np.sqrt(
        np.maximum(node_potential[:, None] * node_potential[None, :], 0.0)
    )

    proximity = np.zeros((n, n), dtype=float)
    proximity[valid] = np.exp(
        -np.minimum(safe[valid] / scale, 60.0)
    )

    score = (
        0.42 * inverse_scaled
        + 0.20 * mutual_competition
        + 0.16 * local_exclusivity
        + 0.12 * rank_score
        + 0.10 * endpoint_potential
    )

    score *= 0.85 + 0.15 * proximity
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    valid_count = float(np.count_nonzero(valid))
    exploration_floor = 0.05 * valid.astype(float) / max(valid_count, 1.0)
    score = 0.95 * score + exploration_floor

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    return (score / maximum).astype(float, copy=False)
