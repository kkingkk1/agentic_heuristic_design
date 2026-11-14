import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    fallback = float(np.median(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    safe_d = np.where(valid, d, fallback * 10.0)
    eps = np.finfo(float).eps
    safe_d = np.maximum(safe_d, eps)

    local_scale = np.partition(safe_d, 1, axis=1)[:, 1]
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = safe_d / np.maximum(geometric_scale, eps)

    distance_score = np.exp(-0.85 * normalized_distance)

    scale_sum = local_scale[:, None] + local_scale[None, :]
    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / np.maximum(scale_sum, eps)
    scale_compatibility = 1.0 / (1.0 + scale_ratio)

    score = distance_score * scale_compatibility
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
