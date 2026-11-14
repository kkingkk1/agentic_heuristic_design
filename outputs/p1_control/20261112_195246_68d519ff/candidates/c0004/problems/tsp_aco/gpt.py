import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = distance_matrix.copy()
    finite_positive = np.isfinite(d) & (d > 0)

    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    replacement = np.max(d[finite_positive])
    d[~np.isfinite(d)] = replacement
    d[d < 0] = replacement

    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    fallback = np.max(positive) if positive.size else 1.0
    d[d <= 0] = fallback

    inv_distance = 1.0 / (d + np.finfo(float).eps)

    local_scale = np.min(d, axis=1)
    local_scale[~np.isfinite(local_scale)] = fallback
    scale_affinity = np.exp(-d / (local_scale[:, None] + np.finfo(float).eps))

    ranks = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    rank_affinity = 1.0 / (1.0 + ranks)

    mutual = np.sqrt(rank_affinity * rank_affinity.T)

    score = (
        0.50 * inv_distance / (np.max(inv_distance, axis=1, keepdims=True) + np.finfo(float).eps)
        + 0.30 * scale_affinity
        + 0.20 * mutual
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
