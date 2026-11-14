import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return lightweight rank-based priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    safe = np.where(valid, d, np.inf)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_gap = np.partition(safe, kth=min(1, n - 1), axis=1)[:, min(1, n - 1)]
    col_gap = np.partition(safe, kth=min(1, n - 1), axis=0)[min(1, n - 1), :]

    row_gap = np.maximum(row_gap - row_best, 0.0)
    col_gap = np.maximum(col_gap - col_best, 0.0)

    row_gap_scale = (
        float(np.median(row_gap[row_gap > 0.0]))
        if np.any(row_gap > 0.0)
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap[col_gap > 0.0]))
        if np.any(col_gap > 0.0)
        else scale
    )

    row_gap_scale = max(row_gap_scale, np.finfo(float).eps)
    col_gap_scale = max(col_gap_scale, np.finfo(float).eps)

    row_regret = np.exp(
        -np.maximum(d - row_best[:, None], 0.0) / row_gap_scale
    )
    col_regret = np.exp(
        -np.maximum(d - col_best[None, :], 0.0) / col_gap_scale
    )
    regret = np.sqrt(row_regret * col_regret)

    row_rank = np.argsort(np.argsort(np.where(valid, d, np.inf), axis=1), axis=1)
    col_rank = np.argsort(np.argsort(np.where(valid, d, np.inf), axis=0), axis=0)

    row_count = np.maximum(valid.sum(axis=1, keepdims=True) - 1, 1)
    col_count = np.maximum(valid.sum(axis=0, keepdims=True) - 1, 1)

    row_rank_score = 1.0 - np.divide(row_rank, row_count)
    col_rank_score = 1.0 - np.divide(col_rank, col_count)
    rank_score = np.sqrt(np.clip(row_rank_score, 0.0, 1.0) *
                         np.clip(col_rank_score, 0.0, 1.0))

    local_scale = max(scale * 0.5, np.finfo(float).eps)
    distance_score = np.exp(
        -np.clip(
            np.maximum(d - row_best[:, None], 0.0) / local_scale,
            0.0,
            50.0,
        )
    )

    score = (
        0.50 * distance_score
        + 0.30 * rank_score
        + 0.20 * regret
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
