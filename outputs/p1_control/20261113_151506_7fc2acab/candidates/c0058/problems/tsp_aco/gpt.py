import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic symmetric edge-priority indicators for a TSP matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distances must be nonnegative")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    # Use a symmetric distance representation for undirected TSP sampling.
    work = 0.5 * (distances + distances.T)
    np.fill_diagonal(work, np.inf)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_edges = work[off_diagonal]
    positive_edges = finite_edges[finite_edges > 0]

    if positive_edges.size:
        scale = float(np.median(positive_edges))
    else:
        scale = 1.0
    scale = max(scale, np.finfo(float).eps)

    normalized = work / scale

    # Robust continuous preference for short edges.
    inverse = 1.0 / (normalized + 1e-9)
    inverse[~np.isfinite(inverse)] = 0.0
    inverse = np.power(np.maximum(inverse, 0.0), 0.75)

    # Rank-based preference at each endpoint.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values

    valid = np.isfinite(work)
    ranks[~valid] = n
    rank_score = 1.0 / np.sqrt(1.0 + ranks)

    # Reward edges that are simultaneously strong from both endpoints.
    bilateral_rank = np.sqrt(rank_score * rank_score.T)

    # Endpoint-specific local contrast: favor edges that beat alternatives.
    sorted_distances = np.sort(np.where(valid, work, np.inf), axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 1 else best

    best[~np.isfinite(best)] = scale
    second[~np.isfinite(second)] = best[~np.isfinite(second)]

    contrast_i = (second[:, None] - work) / (second[:, None] + scale)
    contrast_j = (second[None, :] - work) / (second[None, :] + scale)
    contrast = 1.0 + 0.45 * np.maximum(contrast_i, 0.0)
    contrast *= 1.0 + 0.45 * np.maximum(contrast_j, 0.0)
    contrast[~valid] = 0.0

    # Mutual nearest-neighbor reinforcement.
    nearest = np.argmin(work, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score = inverse
    score *= 0.55 + 0.45 * bilateral_rank
    score *= contrast
    score *= 1.0 + 0.35 * mutual

    # Preserve symmetry and remove invalid self-loops.
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~off_diagonal] = 0.0
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    mean_score = float(np.mean(score[off_diagonal]))
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score
