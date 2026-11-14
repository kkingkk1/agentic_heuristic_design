import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    if not np.any(positive):
        return np.zeros_like(d, dtype=float)

    safe_d = np.where(positive, d, np.inf)
    scale = np.median(d[positive])
    scale = max(scale, np.finfo(float).eps)

    affinity = np.exp(-safe_d / scale)
    affinity[~positive] = 0.0

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        order = np.argsort(safe_d[i])
        valid = positive[i, order]
        order = order[valid]
        ranks[i, order] = np.arange(1, len(order) + 1)

    rank_score = np.exp(-(ranks - 1.0) / max(1.0, np.sqrt(n)))
    rank_score[ranks >= n] = 0.0

    mutual_bonus = np.sqrt(rank_score * rank_score.T)
    score = affinity * (0.65 + 0.35 * mutual_bonus)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float)
