import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic edge-priority indicators for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros_like(distances, dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        raise ValueError("distance_matrix must contain only finite values")

    if np.any(distances < 0):
        raise ValueError("distances must be nonnegative")

    result = np.zeros((n, n), dtype=float)
    if n == 1:
        return result

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Robust inverse-distance preference.
    normalized = work / scale
    inverse = 1.0 / (normalized + 1e-12)
    inverse[~np.isfinite(inverse)] = 0.0

    # Convert each row into a soft local preference distribution.
    row_finite = np.isfinite(work)
    row_min = np.min(np.where(row_finite, work, np.inf), axis=1)
    row_min[~np.isfinite(row_min)] = 0.0
    row_regret = np.maximum(work - row_min[:, None], 0.0)
    local_competition = 1.0 / (1.0 + row_regret / scale)
    local_competition[~row_finite] = 0.0

    # Emphasize edges that are especially competitive from both endpoints.
    score = inverse * np.sqrt(local_competition * local_competition.T)

    # Bonus for mutual nearest-neighbor structure.
    nearest = np.argmin(work, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score *= 1.0 + 0.5 * mutual
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    mean_score = float(np.mean(score[off_diagonal]))
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score
