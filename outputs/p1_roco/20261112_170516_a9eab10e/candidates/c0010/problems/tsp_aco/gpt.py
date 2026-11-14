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

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    positive = finite & (d > 0)

    fallback = np.max(d[positive]) if np.any(positive) else 1.0
    safe_d = np.where(finite, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, np.finfo(float).eps)

    positive_values = safe_d[positive]
    scale = np.median(positive_values) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse = 1.0 / (1.0 + safe_d / scale)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    rank_score = 1.0 - ranks / max(n - 1, 1)

    local_scale = np.partition(safe_d, 0, axis=1)[:, None]
    local_scale = np.maximum(local_scale, np.finfo(float).eps)
    relative_score = 1.0 / (1.0 + safe_d / local_scale)

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    regret = np.partition(safe_d, 1, axis=1)[:, 1] - np.partition(safe_d, 0, axis=1)[:, 0]
    regret = regret / (regret.mean() + np.finfo(float).eps)
    endpoint_importance = 1.0 + 0.25 * np.minimum(regret, 4.0)
    importance = np.sqrt(endpoint_importance @ endpoint_importance.T)

    score = (
        inverse ** 0.45
        * rank_score ** 0.25
        * relative_score ** 0.20
        * (1.0 + 0.35 * mutual) ** 0.10
        * importance
    )

    score[~positive] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
