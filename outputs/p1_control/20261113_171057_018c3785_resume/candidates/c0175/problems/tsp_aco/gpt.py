import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
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

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    positive = d[valid & (d > 0.0)]
    base_scale = (
        float(np.median(positive))
        if positive.size
        else float(np.median(d[valid]))
    )
    base_scale = max(base_scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    row_has_edge = np.isfinite(row_best)
    col_has_edge = np.isfinite(col_best)

    row_best_safe = np.where(row_has_edge, row_best, 0.0)
    col_best_safe = np.where(col_has_edge, col_best, 0.0)

    row_second = np.partition(safe, 1, axis=1)[:, 1]
    col_second = np.partition(safe, 1, axis=0)[1, :]

    row_gap = np.where(
        row_has_edge & np.isfinite(row_second),
        np.maximum(row_second - row_best, 0.0),
        0.0,
    )
    col_gap = np.where(
        col_has_edge & np.isfinite(col_second),
        np.maximum(col_second - col_best, 0.0),
        0.0,
    )

    positive_row_gap = row_gap[row_gap > 0.0]
    positive_col_gap = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(positive_row_gap))
        if positive_row_gap.size
        else base_scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gap))
        if positive_col_gap.size
        else base_scale
    )

    eps = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_delta = np.where(
        valid,
        np.maximum(d - row_best_safe[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_best_safe[None, :], 0.0),
        0.0,
    )

    temperature = max(0.25 * base_scale, eps)

    affinity = np.exp(-np.minimum(row_delta / temperature, 60.0))
    affinity *= np.exp(-0.35 * np.minimum(col_delta / temperature, 60.0))
    affinity[~valid] = 0.0

    row_regret = np.exp(-row_delta / row_gap_scale)
    col_regret = np.exp(-col_delta / col_gap_scale)
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    balanced = affinity.copy()

    for _ in range(32):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

    score = balanced * (0.35 + 0.65 * np.power(regret, 0.8))
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
