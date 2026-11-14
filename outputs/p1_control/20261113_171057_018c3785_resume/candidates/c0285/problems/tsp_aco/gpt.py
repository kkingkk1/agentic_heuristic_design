import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized regret-aware edge priors for a directed TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    tiny = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, tiny)

    safe_d = np.where(valid, d, np.inf)

    row_has = valid.any(axis=1)
    col_has = valid.any(axis=0)

    row_min = np.min(safe_d, axis=1, where=valid, initial=np.inf)
    col_min = np.min(safe_d, axis=0, where=valid, initial=np.inf)

    row_min = np.where(np.isfinite(row_min), row_min, scale)
    col_min = np.where(np.isfinite(col_min), col_min, scale)

    # Partial selection is cheaper than sorting every row and column.
    row_two = np.partition(safe_d, kth=1, axis=1)[:, :2]
    col_two = np.partition(safe_d, kth=1, axis=0)[:2, :]

    row_second = row_two[:, 1]
    col_second = col_two[1, :]

    row_second = np.where(np.isfinite(row_second), row_second, scale)
    col_second = np.where(np.isfinite(col_second), col_second, scale)

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    row_gap_values = row_gap[row_gap > 0.0]
    col_gap_values = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else scale
    )

    if not np.isfinite(row_gap_scale) or row_gap_scale <= 0.0:
        row_gap_scale = scale
    if not np.isfinite(col_gap_scale) or col_gap_scale <= 0.0:
        col_gap_scale = scale

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)
    row_delta = np.where(valid, row_delta, 0.0)
    col_delta = np.where(valid, col_delta, 0.0)

    row_regret = np.exp(
        -np.clip(row_delta / max(row_gap_scale, tiny), 0.0, 60.0)
    )
    col_regret = np.exp(
        -np.clip(col_delta / max(col_gap_scale, tiny), 0.0, 60.0)
    )
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.42 * scale, tiny)
    local_affinity = np.exp(
        -np.clip(row_delta / temperature, 0.0, 60.0)
    )
    local_affinity[~valid] = 0.0

    balanced = local_affinity.copy()

    for _ in range(6):
        row_sum = balanced.sum(axis=1, keepdims=True)
        np.divide(
            balanced,
            row_sum,
            out=balanced,
            where=np.isfinite(row_sum) & (row_sum > 0.0),
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        np.divide(
            balanced,
            col_sum,
            out=balanced,
            where=np.isfinite(col_sum) & (col_sum > 0.0),
        )

        balanced[~np.isfinite(balanced)] = 0.0

    combined = 0.55 * balanced + 0.45 * local_affinity
    score = combined * np.power(np.maximum(regret, 0.0) + tiny, 0.8)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
