import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distances.copy()
    d[~np.isfinite(d)] = np.inf
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d)

    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0
    d[~finite] = fallback * 10.0

    scale = np.median(d[np.isfinite(d) & (d > 0)])
    if not np.isfinite(scale) or scale <= 0:
        scale = fallback
    scale = max(scale, np.finfo(float).eps)

    # Smooth inverse-distance preference.
    inverse = 1.0 / (d + 0.02 * scale)

    # Rank preference: the closest edges from each node receive larger scores.
    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.5, 0.22 * n))

    # Prefer edges that are simultaneously attractive from both endpoints.
    row_best = np.min(d, axis=1)
    mutuality = np.sqrt(
        (row_best[:, None] / (d + scale * 1e-12))
        * (row_best[None, :] / (d + scale * 1e-12))
    )

    # Candidate-list gate retains exploration while penalizing extreme outliers.
    row_median = np.median(d, axis=1)
    candidate_gate = 1.0 / (
        1.0 + np.maximum(0.0, d - row_median[:, None]) / (scale + 1e-12)
    )

    score = (
        0.42 * inverse / (np.mean(inverse, axis=1, keepdims=True) + 1e-12)
        + 0.28 * rank_score
        + 0.20 * mutuality
        + 0.10 * candidate_gate
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
