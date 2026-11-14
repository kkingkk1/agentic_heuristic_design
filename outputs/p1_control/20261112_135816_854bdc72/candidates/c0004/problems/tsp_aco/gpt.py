import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-promise indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distances, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    inverse = 1.0 / (d + eps)
    inverse[~np.isfinite(inverse)] = 0.0
    np.fill_diagonal(inverse, 0.0)

    if n > 1:
        order = np.argsort(d, axis=1)
        ranks = np.empty_like(order, dtype=float)
        ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
        rank_score = 1.0 / (1.0 + ranks)
        rank_score[:, np.arange(n)] = 0.0

        nearest = order[:, 0]
        mutual = np.zeros((n, n), dtype=float)
        for i in range(n):
            j = nearest[i]
            if j != i and nearest[j] == i:
                mutual[i, j] = 1.0
                mutual[j, i] = 1.0

        score = (
            0.55 * inverse / (np.max(inverse) + eps)
            + 0.30 * rank_score
            + 0.15 * mutual
        )
    else:
        score = np.zeros((n, n), dtype=float)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    finite_score = score[np.isfinite(score) & (score > 0)]
    if finite_score.size:
        score /= np.mean(finite_score)

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
