import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return locally scaled edge priorities for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, eps)

    positive_distances = np.where(valid & (d > 0.0), d, np.inf)

    k = min(3, n - 1)
    nearest = np.partition(positive_distances, kth=k - 1, axis=1)[:, :k]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)

    endpoint_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    safe_d = np.where(valid, np.maximum(d, eps), np.inf)
    normalized = safe_d / endpoint_scale

    affinity = 1.0 / (1.0 + normalized)

    median_scale = max(float(np.median(local_scale)), eps)
    sparse_bonus = np.sqrt(
        np.clip(local_scale[:, None] / median_scale, 0.5, 2.0)
        * np.clip(local_scale[None, :] / median_scale, 0.5, 2.0)
    )

    score = affinity * (0.85 + 0.15 * sparse_bonus)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    return np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    ).astype(float, copy=False)
