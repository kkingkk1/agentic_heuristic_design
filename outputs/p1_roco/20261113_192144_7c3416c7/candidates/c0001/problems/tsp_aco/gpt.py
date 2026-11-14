import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diag = ~np.eye(n, dtype=bool)
    finite_values = d[off_diag & np.isfinite(d)]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0]
    scale = np.median(positive_values) if positive_values.size else 1.0
    scale = max(float(scale), np.finfo(float).eps)

    finite_d = np.where(np.isfinite(d), d, 10.0 * scale)
    positive_d = np.maximum(finite_d, np.finfo(float).eps)

    inverse_score = 1.0 / positive_d
    inverse_score /= np.max(inverse_score[off_diag])

    exponential_score = np.exp(-finite_d / scale)
    exponential_score[~off_diag] = np.minimum(
        exponential_score[~off_diag], 1.0
    )

    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(finite_d[i] + np.eye(n)[i] * np.inf)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        rank_score[i] = 1.0 - ranks / max(n - 1, 1)

    rank_score = 0.5 * (rank_score + rank_score.T)
    np.fill_diagonal(rank_score, 0.0)

    nearest = np.partition(
        np.where(np.eye(n, dtype=bool), np.inf, finite_d),
        kth=min(2, n - 1),
        axis=1,
    )[:, : min(2, n - 1)]
    reference = np.mean(nearest, axis=1)

    savings = reference[:, None] + reference[None, :] - finite_d
    savings = np.maximum(savings, 0.0)
    max_savings = np.max(savings[off_diag])
    if max_savings > 0:
        savings /= max_savings

    score = (
        0.40 * inverse_score
        + 0.25 * exponential_score
        + 0.20 * rank_score
        + 0.15 * savings
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float)
