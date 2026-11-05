import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    valid = np.isfinite(d) & (d > 0)
    positive = d[valid]
    scale = np.median(positive) if positive.size else 1.0

    affinity = np.zeros_like(d)
    affinity[valid] = np.exp(-d[valid] / (scale + 1e-12))

    rank = np.full_like(d, n, dtype=float)
    masked = np.where(valid, d, np.inf)
    order = np.argsort(masked, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n)
    rank[valid] = ranks[valid]

    rank_bonus = np.zeros_like(d)
    rank_bonus[valid] = 1.0 / (1.0 + rank[valid])

    nearest = np.argmin(masked, axis=1)
    mutual = np.zeros_like(d)
    for i, j in enumerate(nearest):
        if nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score = affinity * (1.0 + 0.8 * rank_bonus + 1.2 * mutual)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = score.max()
    return score / maximum if maximum > 0 else score
