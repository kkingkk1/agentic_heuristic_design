import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = distance_matrix.copy()
    finite = np.isfinite(d)

    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    positive = finite & (d > 0)
    if np.any(positive):
        fallback = np.max(d[positive])
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1.0
    else:
        fallback = 1.0

    d[~finite] = fallback * 10.0
    d[d < 0] = fallback * 10.0
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    safe_d = np.where(valid, d, fallback * 10.0)

    positive_values = safe_d[valid & (safe_d > 0)]
    scale = np.median(positive_values) if positive_values.size else 1.0
    scale = max(float(scale), np.finfo(float).eps)

    inverse_distance = np.exp(-safe_d / scale)
    inverse_distance[~valid] = 0.0

    row_sorted = np.sort(np.where(valid, safe_d, np.inf), axis=1)
    kth = row_sorted[:, min(2, max(n - 1, 0))]
    kth = np.where(np.isfinite(kth), kth, scale)

    local_competitiveness = np.exp(-safe_d / (kth[:, None] + scale * 0.25))
    local_competitiveness *= np.exp(-safe_d / (kth[None, :] + scale * 0.25))
    local_competitiveness[~valid] = 0.0

    score = 0.65 * inverse_distance + 0.35 * local_competitiveness
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
