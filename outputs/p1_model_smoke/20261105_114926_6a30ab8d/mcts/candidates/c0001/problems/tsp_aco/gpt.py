import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    positive = valid & (~np.eye(n, dtype=bool))

    finite_values = d[positive]
    fallback = float(np.max(finite_values)) if finite_values.size else 1.0
    safe_d = np.where(positive, d, fallback * 10.0 + 1.0)

    safe_d[np.eye(n, dtype=bool)] = np.inf

    row_scale = np.median(
        np.where(np.isfinite(safe_d), safe_d, np.nan),
        axis=1
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        np.maximum(np.mean(np.where(np.isfinite(safe_d), safe_d, 0.0), axis=1), 1.0)
    )

    distance_score = np.exp(-safe_d / row_scale[:, None])

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)

    nearest = order[:, :max(1, min(n - 1, 5))]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        mutual[i, nearest[i]] = 1.0
    mutual = mutual * mutual.T

    score = 0.55 * distance_score + 0.30 * rank_score + 0.15 * mutual
    score[~positive] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
