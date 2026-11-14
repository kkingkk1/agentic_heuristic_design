import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0)

    safe_distances = np.where(positive, distances, np.inf)
    safe_distances = np.minimum(safe_distances, safe_distances.T)

    valid_values = safe_distances[np.isfinite(safe_distances)]
    if valid_values.size == 0:
        return np.zeros_like(distances, dtype=float)

    scale = float(np.median(valid_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    inverse = np.zeros((n, n), dtype=float)
    inverse[positive] = 1.0 / (distances[positive] + np.finfo(float).eps)
    inverse = np.minimum(inverse, inverse.T)
    inverse /= max(float(np.max(inverse)), np.finfo(float).eps)

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = distances[i].copy()
        row[~positive[i]] = np.inf
        order = np.argsort(row, kind="stable")
        rank = 0
        for j in order:
            if np.isfinite(row[j]):
                ranks[i, j] = 1.0 / (1.0 + rank)
                rank += 1
    ranks = np.minimum(ranks, ranks.T)
    if np.max(ranks) > 0:
        ranks /= np.max(ranks)

    local = np.exp(-safe_distances / scale)
    local[~np.isfinite(local)] = 0.0
    np.fill_diagonal(local, 0.0)
    local = np.minimum(local, local.T)
    if np.max(local) > 0:
        local /= np.max(local)

    nearest = np.zeros(n, dtype=bool)
    for i in range(n):
        candidates = np.flatnonzero(np.isfinite(safe_distances[i]))
        if candidates.size:
            nearest[i] = True

    mutual = np.zeros((n, n), dtype=float)
    nearest_index = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(np.isfinite(safe_distances[i]))
        if candidates.size:
            nearest_index[i] = candidates[np.argmin(safe_distances[i, candidates])]
    for i in range(n):
        j = nearest_index[i]
        if j >= 0 and nearest_index[j] == i:
            mutual[i, j] = 1.0
    mutual = np.maximum(mutual, mutual.T)

    score = (
        0.45 * inverse
        + 0.25 * ranks
        + 0.20 * local
        + 0.10 * mutual
    )

    score[~(positive & positive.T)] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    return np.maximum(score, 0.0)
