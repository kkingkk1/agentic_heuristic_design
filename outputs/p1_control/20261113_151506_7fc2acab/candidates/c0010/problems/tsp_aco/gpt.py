import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros_like(d, dtype=float)

    fallback = np.max(d[finite]) if finite.any() else 1.0
    fallback = max(float(fallback), 1.0)

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps

    positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    row_scale = np.median(
        np.where(np.isfinite(d), d, np.nan),
        axis=1
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > eps),
        row_scale,
        global_scale
    )

    normalized_distance = d / np.sqrt(row_scale[:, None] * row_scale[None, :])
    distance_score = np.exp(-normalized_distance)

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)

    mutual_neighbor = (
        (ranks < min(4, n - 1)) &
        (ranks.T < min(4, n - 1))
    ).astype(float)

    local_score = np.sqrt(rank_score * rank_score.T)

    score = (
        0.55 * distance_score +
        0.30 * local_score +
        0.15 * mutual_neighbor
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
