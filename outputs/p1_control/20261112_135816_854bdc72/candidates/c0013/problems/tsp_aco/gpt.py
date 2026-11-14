import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive edge-promising indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    x = d.copy()
    finite = np.isfinite(x)
    positive = finite & (x > 0)

    fallback = np.median(x[positive]) if np.any(positive) else 1.0
    x[~finite] = fallback * 10.0
    x[x < 0] = 0.0
    np.fill_diagonal(x, np.inf)

    k = min(max(2, int(np.sqrt(n))), n - 1)
    local_scale = np.partition(x, k - 1, axis=1)[:, k - 1]
    local_scale[~np.isfinite(local_scale)] = fallback
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    temperature = np.sqrt(local_scale[:, None] * local_scale[None, :])
    distance_score = np.exp(-x / temperature)

    order = np.argsort(x, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks = np.maximum(ranks, 1.0)

    rank_score = 1.0 / np.sqrt(ranks[:, None] * ranks[None, :])

    nearest = order[:, 0]
    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i:
            mutual_bonus[i, j] = 1.0

    score = distance_score * (0.7 + 0.3 * rank_score)
    score += 0.15 * mutual_bonus
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
