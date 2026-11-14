import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    positive_values = d[finite_positive]

    if positive_values.size == 0:
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    scale = np.median(positive_values) + eps

    work = np.where(finite_positive, d, np.inf)
    np.fill_diagonal(work, np.inf)

    nearest = np.min(work, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, scale)

    row_typical = np.median(
        np.where(np.isfinite(work), work, np.nan),
        axis=1
    )
    row_typical = np.where(
        np.isfinite(row_typical) & (row_typical > 0),
        row_typical,
        scale
    )

    col_typical = np.median(
        np.where(np.isfinite(work), work, np.nan),
        axis=0
    )
    col_typical = np.where(
        np.isfinite(col_typical) & (col_typical > 0),
        col_typical,
        scale
    )

    valid = np.isfinite(work)
    normalized = work / (
        np.sqrt(row_typical[:, None] * col_typical[None, :]) + eps
    )

    closeness = np.exp(-normalized)
    endpoint_locality = np.exp(
        -0.5 * (
            work / (nearest[:, None] + eps)
            + work / (nearest[None, :] + eps)
        )
    )

    ranks = np.zeros_like(work, dtype=float)
    for i in range(n):
        order = np.argsort(work[i])
        ranks[i, order] = np.arange(1, n + 1, dtype=float)

    rank_score = 1.0 / (ranks + 1.0)
    rank_score[~valid] = 0.0

    score = (
        0.45 * closeness
        + 0.35 * endpoint_locality
        + 0.20 * rank_score
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
