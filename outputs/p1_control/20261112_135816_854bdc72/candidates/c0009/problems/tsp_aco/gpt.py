import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(distances, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    positive = finite & (d > 0.0)

    if np.any(positive):
        scale = np.median(d[positive])
        if not np.isfinite(scale) or scale <= 0.0:
            scale = np.mean(d[positive])
    else:
        scale = 1.0

    scale = max(float(scale), np.finfo(float).eps)
    similarity = np.zeros_like(d)
    similarity[finite] = 1.0 / (1.0 + d[finite] / scale)

    rank_score = np.zeros_like(d)
    rarity_score = np.zeros_like(d)

    for i in range(n):
        valid = np.isfinite(d[i])
        if not np.any(valid):
            continue

        row_distances = d[i, valid]
        order = np.argsort(row_distances, kind="stable")
        ranks = np.empty(order.size, dtype=float)
        ranks[order] = np.arange(order.size, dtype=float)

        count = max(order.size - 1, 1)
        rank_score[i, valid] = 1.0 - ranks / count

        best = row_distances[order[0]]
        second = row_distances[order[1]] if order.size > 1 else best
        gap = max(second - best, 0.0) / (second + scale)
        rarity_score[i, valid] = gap * rank_score[i, valid]

    score = (
        0.50 * similarity
        + 0.30 * rank_score
        + 0.20 * 0.5 * (rarity_score + rarity_score.T)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
