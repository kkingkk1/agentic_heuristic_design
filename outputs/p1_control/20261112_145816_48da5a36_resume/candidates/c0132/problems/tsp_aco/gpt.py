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

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d > 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    fallback = float(np.median(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, np.finfo(float).eps)

    neighbor_index = min(2, n - 2)
    local_scale = np.partition(
        safe_d, neighbor_index, axis=1
    )[:, neighbor_index]
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    global_scale = float(np.median(local_scale))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = fallback

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = safe_d / geometric_scale
    distance_score = 1.0 / (1.0 + normalized_distance)

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks /= max(n - 1, 1)

    rank_score = 1.0 / (
        1.0 + 0.75 * (ranks + ranks.T)
    )

    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + np.finfo(float).eps
    )
    scale_compatibility = 1.0 / (1.0 + scale_ratio)

    density_penalty = 1.0 / (
        1.0 + 0.15 * np.sqrt(
            global_scale / (geometric_scale + np.finfo(float).eps)
        )
    )

    score = (
        distance_score
        * rank_score
        * scale_compatibility
        * density_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
