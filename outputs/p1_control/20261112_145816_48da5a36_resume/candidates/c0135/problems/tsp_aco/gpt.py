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

    positive = d[valid]
    fallback = float(np.median(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    eps = np.finfo(float).eps
    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, eps)

    # Use the second-nearest valid scale without row-wise sorting.
    kth = min(1, n - 1)
    local_scale = np.partition(safe_d, kth, axis=1)[:, kth]

    # Rows with fewer than two valid edges use their nearest valid edge.
    valid_count = valid.sum(axis=1)
    nearest = np.min(
        np.where(valid, safe_d, np.inf),
        axis=1,
    )
    local_scale = np.where(
        valid_count >= 2,
        local_scale,
        np.where(np.isfinite(nearest), nearest, fallback),
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = safe_d / np.maximum(geometric_scale, eps)

    short_edge_score = np.exp(-0.85 * normalized_distance)
    continuous_rank_proxy = 1.0 / (1.0 + 0.75 * normalized_distance)

    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )
    scale_compatibility = np.exp(-0.20 * scale_ratio)

    score = (
        short_edge_score
        * continuous_rank_proxy
        * scale_compatibility
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
