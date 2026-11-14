import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    work = np.where(valid, d, np.inf)

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]

    if n > 2:
        second = sorted_distances[:, 1]
        third = sorted_distances[:, 2]
    else:
        second = best.copy()
        third = best.copy()

    finite_best = np.isfinite(best)
    safe_best = np.where(finite_best, best, 1.0)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)
    safe_third = np.where(np.isfinite(third), third, 2.0 * safe_second)

    eps = np.finfo(float).eps

    local_scale = np.maximum(
        (safe_best + safe_second + safe_third) / 3.0,
        eps,
    )

    regret = np.clip(
        (safe_second - safe_best) / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )

    urgency = 1.0 + 0.8 * regret

    normalized = np.divide(
        d,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(-np.clip(normalized, 0.0, 60.0))

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 60.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 60.0))

    mutual_quality = np.sqrt(row_quality * col_quality)

    row_rank = np.argsort(np.argsort(np.where(valid, d, np.inf), axis=1), axis=1)
    col_rank = np.argsort(np.argsort(np.where(valid, d, np.inf), axis=0), axis=0)

    rank_quality = 1.0 / (
        1.0 + row_rank.astype(float) + col_rank.astype(float)
    )

    score = (
        distance_quality
        * mutual_quality
        * rank_quality
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0,
    )
