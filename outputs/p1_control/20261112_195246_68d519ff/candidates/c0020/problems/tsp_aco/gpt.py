import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum((d + d.T) / 2.0, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = np.median(positive) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_scale = np.median(
        np.where(np.isfinite(d), d, np.nan),
        axis=1
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        scale
    )

    local_distance = d / np.sqrt(row_scale[:, None] * row_scale[None, :])
    distance_score = np.exp(-local_distance)

    inverse_score = 1.0 / (1.0 + d / scale)
    inverse_score[~np.isfinite(inverse_score)] = 0.0

    ranks = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))

    nearest = np.argmin(d, axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual_bonus[i, j] = 1.0
            mutual_bonus[j, i] = 1.0

    score = (
        0.45 * distance_score
        + 0.30 * inverse_score
        + 0.20 * rank_score
        + 0.05 * mutual_bonus
    )

    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = score.max()
    if maximum > 0:
        score /= maximum

    return score.astype(float)
