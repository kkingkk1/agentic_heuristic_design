import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    if not np.any(positive):
        return np.zeros_like(d, dtype=float)

    values = d[positive]
    scale = np.median(values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_d = np.where(positive, d, np.inf)

    inv = np.zeros_like(d, dtype=float)
    inv[positive] = 1.0 / (1.0 + d[positive] / scale)

    ranks = np.zeros_like(d, dtype=float)
    for i in range(n):
        order = np.argsort(safe_d[i])
        valid = np.isfinite(safe_d[i, order])
        order = order[valid]
        if order.size:
            ranks[i, order] = np.linspace(1.0, 0.15, order.size)

    nearest_count = max(1, min(3, n - 1))
    nearest = np.zeros_like(d, dtype=float)
    for i in range(n):
        order = np.argsort(safe_d[i])
        order = order[np.isfinite(safe_d[i, order])][:nearest_count]
        nearest[i, order] = 1.0

    mutual = nearest * nearest.T
    structural = 0.5 * (ranks + ranks.T)
    proximity = 0.5 * (inv + inv.T)

    score = (
        0.50 * proximity
        + 0.25 * structural
        + 0.15 * mutual
        + 0.10 * (nearest + nearest.T) / 2.0
    )

    score[~np.isfinite(score)] = 0.0
    score[~positive & ~positive.T] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
