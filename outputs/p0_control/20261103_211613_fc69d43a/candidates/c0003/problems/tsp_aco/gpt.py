import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, scale)

    closeness = np.exp(-work / (scale + eps))
    closeness[~np.isfinite(closeness)] = 0.0

    order = np.argsort(work, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for row in range(n):
        ranks[row, order[row]] = rank_values

    rank_score = 1.0 - ranks / max(1.0, n - 1.0)
    rank_score[~np.isfinite(rank_score)] = 0.0
    np.fill_diagonal(rank_score, 0.0)

    sorted_distances = np.take_along_axis(work, order, axis=1)
    second_best = sorted_distances[:, 1] if n > 2 else sorted_distances[:, 0]
    second_best = np.where(np.isfinite(second_best), second_best, scale)

    regret = (second_best[:, None] - work) / (second_best[:, None] + scale + eps)
    regret = np.clip(regret, 0.0, 1.0)
    regret[~np.isfinite(regret)] = 0.0

    endpoint_preference = 0.5 * (rank_score + rank_score.T)
    endpoint_regret = 0.5 * (regret + regret.T)
    symmetric_closeness = 0.5 * (closeness + closeness.T)
    mutual_preference = rank_score * rank_score.T

    scores = (
        0.35 * symmetric_closeness
        + 0.30 * endpoint_preference
        + 0.25 * endpoint_regret
        + 0.10 * mutual_preference
    )

    np.fill_diagonal(scores, 0.0)
    return scores
