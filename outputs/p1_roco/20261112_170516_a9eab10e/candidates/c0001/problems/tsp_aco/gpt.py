import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()
    finite = np.isfinite(d)

    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    max_finite = np.max(d[finite])
    replacement = max_finite + max(1.0, abs(max_finite))
    d[~finite] = replacement
    np.fill_diagonal(d, replacement)

    scale = np.median(d[np.isfinite(d) & (d < replacement)])
    if not np.isfinite(scale) or scale <= 0:
        scale = max(np.mean(d[d < replacement]), 1.0)

    proximity = np.exp(-d / max(scale, np.finfo(float).eps))
    np.fill_diagonal(proximity, 0.0)

    ranks = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    rank_score = 1.0 - ranks / max(n - 2, 1)
    np.fill_diagonal(rank_score, 0.0)
    rank_score = np.clip(rank_score, 0.0, 1.0)

    row_best = np.min(np.where(np.eye(n, dtype=bool), replacement, d), axis=1)
    row_second = np.partition(
        np.where(np.eye(n, dtype=bool), replacement, d), 1, axis=1
    )[:, 1]
    regret = (row_second[:, None] - d) / (
        np.abs(row_second[:, None]) + np.abs(row_best[:, None]) + 1e-12
    )
    regret = np.clip(regret, 0.0, 1.0)
    np.fill_diagonal(regret, 0.0)

    mutual = np.sqrt(rank_score * rank_score.T)
    symmetric_proximity = 0.5 * (proximity + proximity.T)
    symmetric_regret = np.maximum(regret, regret.T)

    score = (
        0.40 * symmetric_proximity
        + 0.30 * mutual
        + 0.20 * symmetric_regret
        + 0.10 * (0.5 * (rank_score + rank_score.T))
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
