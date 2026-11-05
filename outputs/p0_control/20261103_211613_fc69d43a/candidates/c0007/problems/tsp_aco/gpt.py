import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be nonnegative")

    n = d.shape[0]
    scores = np.zeros_like(d, dtype=float)

    if n <= 1:
        return scores

    work = d.copy()
    np.fill_diagonal(work, np.nan)

    row_median = np.nanmedian(work, axis=1)
    row_median = np.maximum(row_median, np.finfo(float).eps)

    relative_distance = work / row_median[:, None]
    closeness = 1.0 / (1.0 + relative_distance)

    row_order = np.argsort(np.where(np.isnan(work), np.inf, work), axis=1)
    rank = np.empty_like(row_order, dtype=float)
    rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    rank_score = 1.0 - rank / max(n - 1, 1)
    rank_score[np.isnan(work)] = 0.0

    mutuality = np.sqrt(np.maximum(rank_score * rank_score.T, 0.0))

    local_contrast = np.exp(
        -0.5 * (
            np.maximum(relative_distance, 0.0)
            + np.maximum(relative_distance.T, 0.0)
        )
    )

    scores = (
        0.45 * np.nan_to_num(closeness, nan=0.0)
        + 0.30 * mutuality
        + 0.25 * np.nan_to_num(local_contrast, nan=0.0)
    )

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = scores.max()
    if maximum > 0:
        scores /= maximum

    return scores
