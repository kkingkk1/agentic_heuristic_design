import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return finite, nonnegative edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    safe_d = np.where(valid, d, np.inf)
    safe_d = np.where(valid & (d > 0.0), d, fallback)
    np.fill_diagonal(safe_d, np.inf)

    available = np.where(np.isfinite(safe_d), safe_d, fallback)
    sorted_distances = np.sort(available, axis=1)
    k = min(3, n - 1)
    local_scale = np.mean(sorted_distances[:, :k], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    normalized_distance = safe_d / local_scale[:, None]

    inverse_affinity = 1.0 / (1.0 + normalized_distance)
    exponential_affinity = np.exp(-normalized_distance)

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    columns = np.arange(n)[None, :]
    ranks[np.arange(n)[:, None], order] = columns
    rank_affinity = 1.0 / (1.0 + ranks.astype(float))
    rank_affinity /= np.max(rank_affinity, axis=1, keepdims=True)

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0

    sparsity = local_scale / (np.mean(local_scale) + np.finfo(float).eps)
    bridge = np.sqrt(sparsity[:, None] * sparsity[None, :])
    bridge = np.clip(bridge, 0.5, 2.0)

    score = (
        0.35 * inverse_affinity
        + 0.25 * exponential_affinity
        + 0.20 * rank_affinity
        + 0.20 * mutual
    )
    score *= 0.8 + 0.2 * bridge

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    result = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
