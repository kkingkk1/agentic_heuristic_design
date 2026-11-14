import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid]
    global_scale = float(np.median(positive))
    global_scale = max(global_scale, np.finfo(float).eps)

    fallback = float(np.max(positive))
    fallback = max(fallback, global_scale)

    safe = np.where(valid, distances, fallback * 10.0)
    safe = 0.5 * (safe + safe.T)
    valid = valid & valid.T
    np.fill_diagonal(valid, False)

    positive_safe = safe[valid]
    if positive_safe.size:
        global_scale = max(
            float(np.median(positive_safe)),
            np.finfo(float).eps,
        )

    valid_counts = valid.sum(axis=1)
    row_sums = np.where(valid, safe, 0.0).sum(axis=1)
    local_scale = np.divide(
        row_sums,
        np.maximum(valid_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=valid_counts > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            np.finfo(float).eps,
        )
    )

    global_term = np.exp(-safe / global_scale)
    local_term = np.exp(-safe / endpoint_scale)

    order = np.argsort(safe, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_term = np.exp(-0.5 * (rank + rank.T) / rank_scale)

    nearest_k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = order[i][valid[i, order[i]]]
        nearest[i, candidates[:nearest_k]] = 1.0

    mutual_term = 1.0 + 0.35 * (nearest * nearest.T)
    one_sided_term = 1.0 + 0.10 * (nearest + nearest.T)

    score = (
        (0.35 + 0.65 * global_term)
        * (0.35 + 0.65 * local_term)
        * (0.55 + 0.45 * rank_term)
        * mutual_term
        * one_sided_term
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
