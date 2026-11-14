import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return low-overhead edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    positive_distances = np.where(valid & (d > 0.0), d, np.inf)

    k = min(3, n - 1)
    nearest = np.partition(positive_distances, kth=k - 1, axis=1)[:, :k]

    finite_nearest = np.isfinite(nearest)
    local_scale = np.where(nearest < np.inf, nearest, fallback)
    local_scale = np.sum(local_scale, axis=1) / np.maximum(
        finite_nearest.sum(axis=1), 1
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    safe_distance = np.where(valid, np.maximum(d, 0.0), np.inf)
    normalized_distance = np.divide(
        safe_distance,
        endpoint_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=np.isfinite(safe_distance),
    )

    affinity = np.exp(-normalized_distance)

    median_scale = max(float(np.median(local_scale)), eps)
    density = np.clip(local_scale / median_scale, 0.5, 2.0)
    sparse_region_bonus = np.sqrt(
        density[:, None] * density[None, :]
    )

    score = affinity * (0.88 + 0.12 * sparse_region_bonus)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
