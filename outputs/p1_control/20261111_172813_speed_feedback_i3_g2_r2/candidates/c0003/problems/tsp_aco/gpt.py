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
    positive = d[(finite) & (d > 0)]

    if positive.size == 0:
        return np.zeros_like(d, dtype=float)

    scale = float(np.median(positive))
    safe_d = np.where(finite, np.maximum(d, 0.0), scale * 1e6)
    np.fill_diagonal(safe_d, np.inf)

    valid = np.isfinite(safe_d)
    valid &= ~np.eye(n, dtype=bool)

    # Scale-free inverse-distance preference.
    inv = np.zeros((n, n), dtype=float)
    inv[valid] = 1.0 / (safe_d[valid] + 1e-12)
    inv /= np.max(inv) + 1e-12

    if n == 1:
        return inv

    # Mutual-nearest-neighbor compatibility.
    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0

    # Local savings: joining i and j is attractive when they are close
    # relative to their respective alternative connections.
    alt_i = np.partition(safe_d, 1, axis=1)[:, 1]
    alt_j = alt_i.copy()
    savings = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j and valid[i, j]:
                baseline = alt_i[i] + alt_j[j]
                savings[i, j] = max(0.0, baseline - safe_d[i, j])

    max_savings = np.max(savings)
    if max_savings > 0:
        savings /= max_savings

    # Reward edges that are short in both directions.
    reciprocal = 0.5 * (inv + inv.T)

    score = (
        0.50 * inv
        + 0.25 * reciprocal
        + 0.15 * savings
        + 0.10 * mutual
    )

    score[~valid] = 0.0

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
