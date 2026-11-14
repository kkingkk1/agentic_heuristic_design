import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return finite, regret-aware edge priors for directed TSP edges."""
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

    safe_d = np.where(valid, d, np.inf)

    positive = safe_d[valid & (safe_d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    tiny = np.finfo(float).eps
    scale = max(scale, tiny)

    row_has = valid.any(axis=1)
    col_has = valid.any(axis=0)

    row_min = np.full(n, scale, dtype=float)
    col_min = np.full(n, scale, dtype=float)

    row_min[row_has] = np.min(safe_d[row_has], axis=1)
    col_min[col_has] = np.min(safe_d[:, col_has], axis=0)

    row_sorted = np.sort(safe_d, axis=1)
    col_sorted = np.sort(safe_d, axis=0)

    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    row_second[row_has] = row_sorted[row_has, 1]
    col_second[col_has] = col_sorted[1, col_has]

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

    if not np.isfinite(row_gap_scale) or row_gap_scale <= 0.0:
        row_gap_scale = scale
    if not np.isfinite(col_gap_scale) or col_gap_scale <= 0.0:
        col_gap_scale = scale

    row_delta = np.maximum(
        np.where(valid, safe_d - row_min[:, None], 0.0),
        0.0,
    )
    col_delta = np.maximum(
        np.where(valid, safe_d - col_min[None, :], 0.0),
        0.0,
    )

    row_regret = np.exp(
        -np.clip(row_delta / row_gap_scale, 0.0, 60.0)
    )
    col_regret = np.exp(
        -np.clip(col_delta / col_gap_scale, 0.0, 60.0)
    )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0
    regret[~np.isfinite(regret)] = 0.0

    temperature = max(scale * 0.42, tiny)
    local_affinity = np.exp(
        -np.clip(row_delta / temperature, 0.0, 60.0)
    )
    local_affinity[~valid] = 0.0
    local_affinity[~np.isfinite(local_affinity)] = 0.0

    balanced = local_affinity.copy()

    for _ in range(48):
        row_sum = np.sum(balanced, axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(row_sum) & (row_sum > 0.0),
        )

        col_sum = np.sum(balanced, axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(col_sum) & (col_sum > 0.0),
        )

        balanced[~np.isfinite(balanced)] = 0.0

    combined = 0.45 * balanced + 0.55 * local_affinity
    score = combined * np.power(regret + tiny, 0.8)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
