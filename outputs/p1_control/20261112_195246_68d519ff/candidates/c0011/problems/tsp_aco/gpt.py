import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    fallback = np.max(d[finite_positive])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.where(finite_positive, d, fallback)
    np.fill_diagonal(clean, fallback)

    positive_values = clean[~np.eye(n, dtype=bool)]
    scale = np.median(positive_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = fallback

    inverse_score = 1.0 / (1.0 + clean / scale)
    np.fill_diagonal(inverse_score, 0.0)

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values

    rank_score = 1.0 - ranks / max(n - 1, 1)
    np.fill_diagonal(rank_score, 0.0)

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    desirability = (
        0.55 * inverse_score
        + 0.30 * rank_score
        + 0.15 * mutual
    )

    desirability = 0.5 * (desirability + desirability.T)
    np.fill_diagonal(desirability, 0.0)

    maximum = np.max(desirability)
    if maximum > 0 and np.isfinite(maximum):
        desirability /= maximum

    return desirability
