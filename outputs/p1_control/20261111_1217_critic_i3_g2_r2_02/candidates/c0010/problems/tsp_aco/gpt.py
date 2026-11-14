import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    if not np.all(finite | np.eye(n, dtype=bool)):
        raise ValueError("distance_matrix contains invalid non-finite off-diagonal values")

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    local_scale = np.quantile(nearest, 0.6, axis=1)
    local_scale = np.maximum(local_scale, eps)

    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    ranks /= max(1.0, n - 1)

    scale_pair = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = work / (scale_pair + eps)

    locality = np.exp(-1.15 * normalized_distance)

    bilateral_preference = np.exp(-1.7 * (ranks + ranks.T))
    bilateral_preference = 0.25 + 0.75 * bilateral_preference

    scale_ratio = local_scale[:, None] / (local_scale[None, :] + eps)
    isolation = np.minimum(scale_ratio, 1.0 / (scale_ratio + eps))
    isolation = np.sqrt(np.clip(isolation, 0.0, 1.0))

    score = locality * (0.60 + 0.40 * bilateral_preference)
    score *= 0.78 + 0.22 * isolation

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
