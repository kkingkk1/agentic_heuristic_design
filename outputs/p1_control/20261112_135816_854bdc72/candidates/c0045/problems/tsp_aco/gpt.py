import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = 0.5 * (d + d.T)
    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0)]
    eps = (
        max(float(np.median(positive)) * 1e-12, np.finfo(float).eps)
        if positive.size
        else np.finfo(float).eps
    )

    fallback = float(np.max(d[valid])) if np.any(valid) else 1.0
    safe_d = np.where(valid, d, fallback + eps)

    row_positive = np.where(valid & (d > 0), d, np.nan)
    local_scale = np.nanmedian(row_positive, axis=1)
    global_scale = float(np.median(positive)) if positive.size else fallback
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        max(global_scale, eps),
    )

    normalized_distance = safe_d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    scale_score = np.exp(-normalized_distance)

    inverse = 1.0 / (safe_d + eps)
    inverse[~valid] = 0.0

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    endpoint_preference = inverse / row_max
    bilateral_preference = np.sqrt(
        endpoint_preference * endpoint_preference.T
    )

    ranks = np.full((n, n), n, dtype=float)
    nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        order = np.argsort(safe_d[i], kind="stable")
        order = order[valid[i, order]]
        ranks[i, order] = np.arange(order.size, dtype=float)
        nearest[i, order[: max(1, min(n - 1, int(np.sqrt(n))))]] = 1.0

    rank_score = np.exp(-0.8 * (ranks + ranks.T) / max(n - 1, 1))
    mutual_neighbor_bonus = nearest * nearest.T

    score = (
        0.40 * scale_score
        + 0.35 * bilateral_preference
        + 0.20 * rank_score
        + 0.05 * mutual_neighbor_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
