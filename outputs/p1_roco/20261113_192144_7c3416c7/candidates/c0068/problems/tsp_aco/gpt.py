import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_off_diagonal = np.isfinite(distances) & off_diagonal
    if not np.any(finite_off_diagonal):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[finite_off_diagonal]
    positive_values = finite_values[finite_values > 0]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(np.isfinite(distances), distances, np.inf)
    safe = np.maximum(safe, 0.0)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + 0.08 * scale)

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(safe[i], kind="stable")
        rank = 1
        for j in order:
            if i == j or not np.isfinite(safe[i, j]):
                continue
            ranks[i, j] = 1.0 / rank
            rank += 1

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            nearest[i] = candidates[np.argmin(safe[i, candidates])]

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    local_scale = np.zeros(n, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            positive_row = safe[i, candidates][safe[i, candidates] > 0]
            if positive_row.size:
                local_scale[i] = float(np.median(positive_row))
            else:
                local_scale[i] = scale

    local_scale[local_scale <= 0] = scale

    proximity = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            proximity[i, candidates] = np.exp(
                -safe[i, candidates] / (1.15 * local_scale[i])
            )

    score = (
        0.56 * inverse * scale
        + 0.22 * ranks
        + 0.14 * proximity
        + 0.08 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~(valid | valid.T)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
