import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)
    if n == 1:
        return np.zeros_like(distances, dtype=float)

    d = np.nan_to_num(distances, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    fallback = np.max(d[finite]) * 2.0 + 1.0
    safe_d = np.where(finite, d, fallback)

    scale = np.median(safe_d[finite])
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(safe_d[finite])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    inverse_distance = 1.0 / (1.0 + safe_d / scale)

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    locality = 1.0 / (1.0 + ranks)

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    mutual[np.arange(n), nearest] = 1.0
    mutual = np.maximum(mutual, mutual.T)

    score = (
        0.60 * inverse_distance
        + 0.25 * (locality + locality.T) / 2.0
        + 0.15 * mutual
    )

    score = (score + score.T) / 2.0
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
