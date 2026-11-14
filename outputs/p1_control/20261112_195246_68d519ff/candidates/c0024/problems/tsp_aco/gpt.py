import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite_offdiag = np.isfinite(d) & (~np.eye(n, dtype=bool))
    positive = finite_offdiag & (d > 0)

    if not np.any(positive):
        return np.zeros_like(d, dtype=float)

    safe_d = np.where(positive, d, np.inf)

    local_scale = np.full(n, np.inf, dtype=float)
    for i in range(n):
        row = safe_d[i]
        valid = np.isfinite(row)
        if np.any(valid):
            local_scale[i] = np.median(row[valid])

    global_scale = np.median(d[positive])
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    normalized = d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    distance_score = np.zeros_like(d, dtype=float)
    distance_score[positive] = np.exp(-normalized[positive])

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        row = safe_d[i].copy()
        row[i] = np.inf
        j = int(np.argmin(row))
        if np.isfinite(row[j]):
            nearest[i] = j

    mutual = np.zeros_like(d, dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    row_rank = np.zeros_like(d, dtype=float)
    for i in range(n):
        valid = positive[i]
        count = int(np.sum(valid))
        if count:
            order = np.argsort(np.where(valid, d[i], np.inf))
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(n, dtype=float)
            row_rank[i, valid] = 1.0 - ranks[valid] / max(count, 1)

    isolation = 1.0 / (1.0 + np.maximum(local_scale, 0.0))
    isolation = isolation / max(np.max(isolation), np.finfo(float).eps)
    isolation_bonus = np.sqrt(isolation[:, None] * isolation[None, :])

    score = (
        0.55 * distance_score
        + 0.25 * np.sqrt(row_rank * row_rank.T)
        + 0.15 * mutual
        + 0.05 * isolation_bonus
    )

    score = np.where(positive, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
