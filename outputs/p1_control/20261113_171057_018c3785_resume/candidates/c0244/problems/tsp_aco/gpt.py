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
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_has = valid.any(axis=1)
    col_has = valid.any(axis=0)

    row_min = np.full(n, scale, dtype=float)
    col_min = np.full(n, scale, dtype=float)
    row_min[row_has] = np.min(
        np.where(valid[row_has], d[row_has], np.inf), axis=1
    )
    col_min[col_has] = np.min(
        np.where(valid[:, col_has], d[:, col_has], np.inf), axis=0
    )

    masked = np.where(valid, d, np.inf)
    row_sorted = np.sort(masked, axis=1)
    col_sorted = np.sort(masked, axis=0)

    row_second = np.where(
        row_has,
        row_sorted[:, 1],
        row_min,
    )
    col_second = np.where(
        col_has,
        col_sorted[1, :],
        col_min,
    )

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)
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

    tiny = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)

    row_regret = np.exp(
        -np.clip(row_delta / row_gap_scale, 0.0, 60.0)
    )
    col_regret = np.exp(
        -np.clip(col_delta / col_gap_scale, 0.0, 60.0)
    )
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    centered = np.where(valid, row_delta, 0.0)
    temperature = max(scale * 0.42, tiny)

    affinity = np.exp(-np.clip(centered / temperature, 0.0, 60.0))
    affinity[~valid] = 0.0

    balanced = affinity.copy()
    for _ in range(48):
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

    score = balanced * np.power(regret + tiny, 0.8)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
