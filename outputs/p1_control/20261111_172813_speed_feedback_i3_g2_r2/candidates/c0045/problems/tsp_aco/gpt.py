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

    finite_positive = np.isfinite(d) & (d > 0)
    positive = d[finite_positive]

    if positive.size == 0:
        return np.zeros_like(d, dtype=float)

    scale = max(float(np.median(positive)), 1e-12)
    fallback = scale * 1e6

    safe_d = np.where(np.isfinite(d) & (d >= 0), d, fallback)
    np.fill_diagonal(safe_d, np.inf)

    valid = np.isfinite(safe_d)
    valid &= ~np.eye(n, dtype=bool)

    if n == 1:
        return np.zeros_like(d, dtype=float)

    safe_d = np.maximum(safe_d, 0.0)

    inv = np.zeros((n, n), dtype=float)
    inv[valid] = 1.0 / (safe_d[valid] + 1e-12)

    inv_max = np.max(inv)
    if inv_max > 0:
        inv /= inv_max

    reciprocal = 0.5 * (inv + inv.T)
    reciprocal_max = np.max(reciprocal)
    if reciprocal_max > 0:
        reciprocal /= reciprocal_max

    nearest = np.argmin(safe_d, axis=1)
    nearest_distance = safe_d[np.arange(n), nearest]

    mutual = (
        valid
        & (nearest[np.arange(n)[:, None]] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    if n > 2:
        two_nearest = np.partition(safe_d, 1, axis=1)[:, :2]
        alternative = np.max(two_nearest, axis=1)
    else:
        alternative = np.full(n, fallback, dtype=float)

    baseline = alternative[:, None] + alternative[None, :]
    savings = np.maximum(0.0, baseline - safe_d)
    savings[~valid] = 0.0

    savings_max = np.max(savings)
    if savings_max > 0:
        savings /= savings_max

    score = (
        0.48 * inv
        + 0.24 * reciprocal
        + 0.18 * savings
        + 0.10 * mutual
    )
    score[~valid] = 0.0

    elite_count = min(
        n - 1,
        max(3, int(np.ceil(np.log2(max(n, 2)))) + 1),
    )

    elite = np.zeros((n, n), dtype=bool)
    if elite_count > 0:
        elite_indices = np.argpartition(
            np.where(valid, score, -np.inf),
            -elite_count,
            axis=1,
        )[:, -elite_count:]
        elite[np.arange(n)[:, None], elite_indices] = True
        elite &= valid

    row_max = np.max(score, axis=1, keepdims=True)
    exploration_floor = 0.025 * row_max

    score = np.where(
        valid,
        np.where(elite, score, np.maximum(0.18 * score, exploration_floor)),
        0.0,
    )

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
