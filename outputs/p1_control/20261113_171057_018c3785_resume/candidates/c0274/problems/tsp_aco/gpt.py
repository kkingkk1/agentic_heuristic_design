import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)
    row_has = valid.any(axis=1)
    col_has = valid.any(axis=0)

    row_best = np.full(n, scale, dtype=float)
    col_best = np.full(n, scale, dtype=float)
    row_best[row_has] = np.min(masked[row_has], axis=1)
    col_best[col_has] = np.min(masked[:, col_has], axis=0)

    row_sorted = np.sort(masked, axis=1)
    col_sorted = np.sort(masked, axis=0)

    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    rows_with_two = valid.sum(axis=1) >= 2
    cols_with_two = valid.sum(axis=0) >= 2

    row_second[rows_with_two] = row_sorted[rows_with_two, 1]
    col_second[cols_with_two] = col_sorted[1, cols_with_two]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_gap[~np.isfinite(row_gap)] = 0.0
    col_gap[~np.isfinite(col_gap)] = 0.0

    positive_row_gaps = row_gap[row_gap > 0.0]
    positive_col_gaps = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(positive_row_gaps))
        if positive_row_gaps.size
        else scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gaps))
        if positive_col_gaps.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_delta = np.where(
        valid,
        np.maximum(d - row_best[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_best[None, :], 0.0),
        0.0,
    )

    row_regret = np.exp(-np.clip(row_delta / row_gap_scale, 0.0, 60.0))
    col_regret = np.exp(-np.clip(col_delta / col_gap_scale, 0.0, 60.0))
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.30 * scale, eps)
    local_affinity = np.exp(
        -np.clip(row_delta / temperature, 0.0, 60.0)
    )
    local_affinity[~valid] = 0.0

    balanced = local_affinity.copy()

    for _ in range(64):
        previous = balanced.copy()

        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(row_sum) & (row_sum > 0.0),
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(col_sum) & (col_sum > 0.0),
        )

        balanced[~np.isfinite(balanced)] = 0.0

        if np.max(np.abs(balanced - previous)) < 1e-10:
            break

    combined = 0.65 * balanced + 0.35 * local_affinity
    score = combined * np.power(regret + eps, 0.9)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
