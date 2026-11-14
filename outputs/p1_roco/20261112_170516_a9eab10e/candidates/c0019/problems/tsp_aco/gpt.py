import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = max(scale * 1e-12, np.finfo(float).eps)

    affinity = 1.0 / (d + eps)
    affinity[~np.isfinite(affinity)] = 0.0

    row_max = np.max(affinity, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    relative_closeness = affinity / row_max

    row_sum = np.sum(affinity, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps)
    scarcity = affinity / row_sum

    rank = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    rank_score = np.exp(-rank / max(1.0, 0.35 * n))

    mutual_closeness = np.sqrt(
        np.maximum(relative_closeness * relative_closeness.T, 0.0)
    )

    neighborhood_size = max(2, min(n - 1, int(np.sqrt(n)) + 1))
    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        nearest = np.argsort(d[i])[:neighborhood_size]
        neighborhood[i, nearest] = 1.0

    shared_neighborhood = (neighborhood @ neighborhood.T) / neighborhood_size

    score = (
        0.42 * mutual_closeness
        + 0.28 * np.sqrt(np.maximum(scarcity * scarcity.T, 0.0))
        + 0.20 * np.sqrt(np.maximum(rank_score * rank_score.T, 0.0))
        + 0.10 * shared_neighborhood
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
