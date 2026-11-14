import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return locally normalized edge-priority indicators for stochastic TSP sampling."""
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

    safe_d = np.where(valid, d, np.inf)
    safe_d = np.where(safe_d > 0.0, safe_d, fallback)
    np.fill_diagonal(safe_d, np.inf)

    k = min(3, n - 1)
    nearest = np.partition(safe_d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    source_affinity = np.exp(-safe_d / local_scale[:, None])
    target_affinity = np.exp(-safe_d / local_scale[None, :])
    reciprocal_affinity = np.sqrt(source_affinity * target_affinity)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1).astype(float)
    rank_affinity = 1.0 / (1.0 + ranks)

    reverse_ranks = ranks.T
    reciprocal_rank = np.sqrt(
        rank_affinity * (1.0 / (1.0 + reverse_ranks))
    )

    score = reciprocal_affinity * (0.8 + 0.2 * reciprocal_rank)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    return np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )
