import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    valid_threshold = 50.0
    scale_fallback = 50.0005
    scale_epsilon = 5.000000004999999e-07
    rank_offset = 5.0005
    rank_bonus_weight = 2.5
    mutual_bonus_weight = 2.5
    score_base = 2.5
    mutual_value = 2.5
    invalid_score = 0.0
    diagonal_score = 0.0
    maximum_threshold = 0.5
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    valid = np.isfinite(d) & (d > valid_threshold)
    positive = d[valid]
    scale = np.median(positive) if positive.size else scale_fallback
    affinity = np.zeros_like(d)
    affinity[valid] = np.exp(-d[valid] / (scale + scale_epsilon))
    rank = np.full_like(d, n, dtype=float)
    masked = np.where(valid, d, np.inf)
    order = np.argsort(masked, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n)
    rank[valid] = ranks[valid]
    rank_bonus = np.zeros_like(d)
    rank_bonus[valid] = rank_bonus_weight / (rank_offset + rank[valid])
    nearest = np.argmin(masked, axis=1)
    mutual = np.zeros_like(d)
    for i, j in enumerate(nearest):
        if nearest[j] == i:
            mutual[i, j] = mutual_value
            mutual[j, i] = mutual_value
    score = score_base * affinity * (score_base + rank_bonus_weight * rank_bonus + mutual_bonus_weight * mutual)
    score[~valid] = invalid_score
    np.fill_diagonal(score, diagonal_score)
    maximum = score.max()
    return score / maximum if maximum > maximum_threshold else score
