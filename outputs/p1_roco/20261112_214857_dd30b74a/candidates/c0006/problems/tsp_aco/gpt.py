import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return prior edge indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    positive_values = d[finite_positive]
    fallback = float(np.max(positive_values))
    safe_d = np.where(finite_positive, d, fallback)
    np.fill_diagonal(safe_d, fallback)

    eps = np.finfo(float).eps

    # Global short-edge preference.
    inv_distance = 1.0 / np.maximum(safe_d, eps)

    # Relative preference from each endpoint.
    row_scale = np.median(safe_d, axis=1, keepdims=True)
    relative_preference = row_scale / np.maximum(safe_d, eps)

    # Clarke-Wright-style savings using a robust virtual depot distance.
    depot_distance = np.median(safe_d, axis=1)
    savings = (
        depot_distance[:, None]
        + depot_distance[None, :]
        - safe_d
    )
    savings = np.maximum(savings, 0.0)

    def robust_scale(x: np.ndarray) -> np.ndarray:
        values = x[np.isfinite(x)]
        if values.size == 0:
            return np.zeros_like(x)
        lo = np.percentile(values, 5.0)
        hi = np.percentile(values, 95.0)
        if hi <= lo + eps:
            return np.ones_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    score = (
        0.50 * robust_scale(inv_distance)
        + 0.30 * robust_scale(relative_preference)
        + 0.20 * robust_scale(savings)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    if n > 1:
        off_diagonal = ~np.eye(n, dtype=bool)
        maximum = np.max(score[off_diagonal])
        if maximum > 0:
            score /= maximum

    return score.astype(float, copy=False)
