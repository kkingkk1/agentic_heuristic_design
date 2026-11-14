import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be nonnegative")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_off_diagonal = np.isfinite(d)
    positive = d[finite_off_diagonal & (d > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(global_scale, 1.0)

    row_scales = np.empty(n, dtype=float)
    for i in range(n):
        row_values = d[i, np.isfinite(d[i]) & (d[i] > 0)]
        row_scales[i] = (
            float(np.median(row_values))
            if row_values.size
            else global_scale
        )
    row_scales = np.maximum(row_scales, eps)

    exponential_preference = np.exp(
        -d / row_scales[:, None]
    )
    exponential_preference[~finite_off_diagonal] = 0.0

    reciprocal_preference = 1.0 / np.maximum(d, eps)
    reciprocal_preference[~finite_off_diagonal] = 0.0

    reciprocal_row_sum = reciprocal_preference.sum(axis=1, keepdims=True)
    reciprocal_preference /= np.maximum(reciprocal_row_sum, eps)

    exponential_row_sum = exponential_preference.sum(axis=1, keepdims=True)
    exponential_preference /= np.maximum(exponential_row_sum, eps)

    mutual_preference = np.sqrt(
        exponential_preference * exponential_preference.T
    )
    mutual_reciprocal = np.sqrt(
        reciprocal_preference * reciprocal_preference.T
    )

    rank_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(d[i], kind="stable")
        rank = 0
        for j in order:
            if i == j or not np.isfinite(d[i, j]):
                continue
            rank_bonus[i, j] = 1.0 / (1.0 + rank)
            rank += 1

    rank_bonus = np.sqrt(rank_bonus * rank_bonus.T)

    score = (
        0.55 * mutual_preference
        + 0.30 * mutual_reciprocal
        + 0.15 * rank_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    return score
