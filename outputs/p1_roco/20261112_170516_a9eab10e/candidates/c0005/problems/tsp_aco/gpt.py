import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0)
    valid &= ~np.eye(n, dtype=bool)

    d = np.where(valid, distance_matrix, np.nan)

    positive = d[d > 0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    local_scale = np.nanmedian(
        np.where((d > 0) & np.isfinite(d), d, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )

    safe_d = np.where(valid, np.maximum(distance_matrix, np.finfo(float).eps), np.inf)

    inverse_score = 1.0 / safe_d
    inverse_score[~valid] = 0.0

    scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    exponential_score = np.exp(-safe_d / np.maximum(scale, np.finfo(float).eps))
    exponential_score[~valid] = 0.0

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        order = np.argsort(safe_d[i], kind="stable")
        rank = 0
        for j in order:
            if valid[i, j]:
                ranks[i, j] = rank
                rank += 1

    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~valid] = 0.0

    nearest = np.argmin(np.where(valid, safe_d, np.inf), axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and valid[i, j] and nearest[j] == i:
            mutual_bonus[i, j] = 1.0

    score = (
        0.40 * inverse_score / np.maximum(np.nanmax(inverse_score, axis=1, keepdims=True), np.finfo(float).eps)
        + 0.30 * exponential_score
        + 0.20 * rank_score
        + 0.10 * mutual_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float)
