import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    positive = valid & (d > 0)
    finite_positive = d[positive]

    if finite_positive.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_positive))
    safe_d = np.where(positive, d, fallback)

    row_scale = np.min(
        np.where(positive, d, np.inf),
        axis=1,
    )
    row_scale = np.where(np.isfinite(row_scale), row_scale, fallback)
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    local = 1.0 / (1.0 + d / np.sqrt(row_scale[:, None] * row_scale[None, :]))

    row_rank = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    col_rank = np.argsort(np.argsort(safe_d, axis=0), axis=0)

    rank_score = 1.0 / (
        1.0
        + row_rank.astype(float)
        + col_rank.astype(float)
    )

    nearest_row = np.argmin(np.where(positive, d, np.inf), axis=1)
    nearest_col = np.argmin(np.where(positive, d, np.inf), axis=0)

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_row[i]
        if 0 <= j < n and nearest_col[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    global_scale = np.median(finite_positive)
    global_scale = max(float(global_scale), np.finfo(float).eps)
    global_score = 1.0 / (1.0 + d / global_scale)

    score = (
        0.50 * local
        + 0.30 * rank_score
        + 0.15 * global_score
        + 0.05 * mutual
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
