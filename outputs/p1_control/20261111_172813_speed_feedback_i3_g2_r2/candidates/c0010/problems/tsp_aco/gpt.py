import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]

    if positive.size == 0:
        return np.zeros_like(d, dtype=float)

    scale = float(np.median(positive))
    fallback = max(scale, 1e-12) * 1e6

    safe_d = np.where(finite, np.maximum(d, 0.0), fallback)
    np.fill_diagonal(safe_d, np.inf)

    valid = np.isfinite(safe_d)
    valid &= ~np.eye(n, dtype=bool)

    inv = np.zeros((n, n), dtype=float)
    inv[valid] = 1.0 / (safe_d[valid] + 1e-12)
    inv_max = np.max(inv)
    if inv_max > 0:
        inv /= inv_max

    if n == 1:
        return inv

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0

    alt_i = np.partition(safe_d, 1, axis=1)[:, 1]
    savings = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if valid[i, j]:
                baseline = alt_i[i] + alt_i[j]
                savings[i, j] = max(0.0, baseline - safe_d[i, j])

    savings_max = np.max(savings)
    if savings_max > 0:
        savings /= savings_max

    reciprocal = 0.5 * (inv + inv.T)

    score = (
        0.45 * inv
        + 0.25 * reciprocal
        + 0.20 * savings
        + 0.10 * mutual
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
