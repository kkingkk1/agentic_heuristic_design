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
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    k = max(1, min(n - 1, int(np.sqrt(n))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)

    invalid_scale = ~np.isfinite(local_scale) | (local_scale <= 0)
    local_scale = np.where(invalid_scale, fallback, local_scale)
    local_scale = np.maximum(local_scale, eps)

    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    ranks /= max(1.0, n - 1)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = work / (pair_scale + eps)

    locality = np.exp(-0.75 * relative_distance)
    bilateral_preference = np.exp(-1.5 * (ranks + ranks.T))

    log_scale_ratio = np.abs(
        np.log((local_scale[:, None] + eps) / (local_scale[None, :] + eps))
    )
    bridge_bonus = 1.0 + 0.20 * np.tanh(log_scale_ratio)

    score = locality
    score *= 0.60 + 0.40 * bilateral_preference
    score *= bridge_bonus

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
