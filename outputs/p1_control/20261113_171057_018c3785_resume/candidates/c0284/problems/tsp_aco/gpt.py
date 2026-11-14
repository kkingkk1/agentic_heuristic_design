import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware edge priors for a directed TSP."""
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

    if not valid.any():
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    tiny = np.finfo(float).eps

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, tiny)

    safe_d = np.where(valid, d, np.inf)

    row_has = valid.any(axis=1)
    col_has = valid.any(axis=0)

    row_min_all = np.min(safe_d, axis=1)
    col_min_all = np.min(safe_d, axis=0)

    row_min = np.where(row_has, row_min_all, scale)
    col_min = np.where(col_has, col_min_all, scale)

    # Partial selection is cheaper than sorting every row and column.
    row_second_all = np.partition(safe_d, 1, axis=1)[:, 1]
    col_second_all = np.partition(safe_d, 1, axis=0)[1, :]

    row_second = np.where(row_has, row_second_all, scale)
    col_second = np.where(col_has, col_second_all, scale)

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)
    row_gap[~np.isfinite(row_gap)] = 0.0
    col_gap[~np.isfinite(col_gap)] = 0.0

    row_positive = row_gap[row_gap > 0.0]
    col_positive = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(row_positive)) if row_positive.size else scale
    )
    col_gap_scale = (
        float(np.median(col_positive)) if col_positive.size else scale
    )

    if not np.isfinite(row_gap_scale) or row_gap_scale <= 0.0:
        row_gap_scale = scale
    if not np.isfinite(col_gap_scale) or col_gap_scale <= 0.0:
        col_gap_scale = scale

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    np.subtract(
        d,
        row_min[:, None],
        out=row_delta,
        where=valid,
    )
    np.subtract(
        d,
        col_min[None, :],
        out=col_delta,
        where=valid,
    )

    np.maximum(row_delta, 0.0, out=row_delta)
    np.maximum(col_delta, 0.0, out=col_delta)

    row_regret = np.exp(
        -np.minimum(row_delta / row_gap_scale, 60.0),
        where=valid,
        out=np.zeros_like(d),
    )
    col_regret = np.exp(
        -np.minimum(col_delta / col_gap_scale, 60.0),
        where=valid,
        out=np.zeros_like(d),
    )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.42 * scale, tiny)
    local_affinity = np.exp(
        -np.minimum(row_delta / temperature, 60.0),
        where=valid,
        out=np.zeros_like(d),
    )
    local_affinity[~valid] = 0.0

    balanced = local_affinity.copy()

    for _ in range(8):
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
    score = combined * np.power(regret + tiny, 0.8)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
