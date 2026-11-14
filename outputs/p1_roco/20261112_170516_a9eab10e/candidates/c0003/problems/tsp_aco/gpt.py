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

    finite = np.isfinite(d)
    off_diag = ~np.eye(n, dtype=bool)
    valid = finite & off_diag

    positive = d[valid & (d > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    safe_d = np.where(valid, d, fallback)
    safe_d = np.maximum(safe_d, np.finfo(float).eps)
    np.fill_diagonal(safe_d, np.inf)

    nearest = np.min(safe_d, axis=1)
    scale = np.maximum(nearest, fallback)

    inverse_score = 1.0 / safe_d
    inverse_score[~valid] = 0.0

    row_rank = safe_d / scale[:, None]
    local_score = 1.0 / (1.0 + row_rank)

    mutual = (
        (safe_d <= nearest[:, None] * 1.5)
        & (safe_d <= nearest[None, :] * 1.5)
        & valid
    ).astype(float)

    neighborhood_similarity = np.exp(
        -np.abs(np.log1p(safe_d / scale[:, None])
                - np.log1p(safe_d / scale[None, :]))
    )
    neighborhood_similarity[~valid] = 0.0

    score = (
        0.55 * inverse_score / np.maximum(np.max(inverse_score), np.finfo(float).eps)
        + 0.20 * local_score
        + 0.20 * mutual
        + 0.05 * neighborhood_similarity
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
