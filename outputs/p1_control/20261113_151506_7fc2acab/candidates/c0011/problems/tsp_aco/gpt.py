import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return prior indicators for promising TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    x = np.array(d, copy=True)
    finite = np.isfinite(x)
    positive = x[finite & (x > 0)]

    if positive.size == 0:
        return np.zeros_like(x, dtype=float)

    fallback = float(np.max(positive))
    x[~finite] = fallback
    x = np.maximum(x, 0.0)
    np.fill_diagonal(x, 0.0)

    off_diag = ~np.eye(n, dtype=bool)
    scale = float(np.median(x[off_diag]))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    closeness = np.exp(-x / (scale + 1e-12))
    np.fill_diagonal(closeness, 0.0)

    ranks = np.empty_like(x)
    for i in range(n):
        order = np.argsort(x[i], kind="stable")
        ranks[i, order] = np.arange(n, dtype=float)
    reciprocal = 1.0 / (1.0 + ranks)
    mutual = np.sqrt(reciprocal * reciprocal.T)
    np.fill_diagonal(mutual, 0.0)

    k = min(max(2, int(np.sqrt(n))), n - 1)
    neighbor_mask = np.zeros((n, n), dtype=float)
    for i in range(n):
        nearest = np.argsort(x[i], kind="stable")[: k + 1]
        nearest = nearest[nearest != i][:k]
        neighbor_mask[i, nearest] = 1.0
    agreement = 0.5 * (neighbor_mask + neighbor_mask.T)

    two_hop = x[:, :, None] + x[None, :, :]
    two_hop = np.min(two_hop + np.eye(n)[None, :, :] * np.inf, axis=1)
    bridge = np.maximum(two_hop - x, 0.0) / (two_hop + x + 1e-12)
    bridge[~np.isfinite(bridge)] = 0.0
    np.fill_diagonal(bridge, 0.0)

    score = (
        0.45 * closeness
        + 0.25 * mutual
        + 0.20 * agreement
        + 0.10 * bridge
    )
    score = 0.5 * (score + score.T)
    score[~off_diag] = 0.0

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
