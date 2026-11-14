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

    finite_positive = np.isfinite(d) & (d > 0)
    safe_d = np.where(finite_positive, d, np.nan)

    row_scale = np.nanmedian(safe_d, axis=1)
    fallback = np.nanmedian(safe_d)
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback,
    )

    local_scale = (row_scale[:, None] + row_scale[None, :]) * 0.5
    relative_distance = np.divide(
        np.maximum(d, 0.0),
        local_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=local_scale > 0,
    )

    distance_score = np.exp(-relative_distance)

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        order = np.argsort(np.where(finite_positive[i], d[i], np.inf))
        valid = finite_positive[i, order]
        ranks[i, order[valid]] = np.arange(1, valid.sum() + 1)

    rank_score = 1.0 / ranks
    mutual_nearest = (
        (ranks == 1)
        & (ranks.T == 1)
        & finite_positive
    )

    score = (
        0.60 * distance_score
        + 0.30 * rank_score
        + 0.10 * mutual_nearest.astype(float)
    )

    score[~finite_positive] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
