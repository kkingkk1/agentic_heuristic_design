import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diversity-preserving, balanced priors for directed TSP edges."""
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

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)
    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_second = np.partition(safe, kth=1, axis=1)[:, 1]
    col_second = np.partition(safe, kth=1, axis=0)[1, :]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

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

    tiny = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_delta = np.where(
        valid,
        np.maximum(d - row_min[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_min[None, :], 0.0),
        0.0,
    )

    temperature = max(0.30 * scale, tiny)
    affinity = np.exp(-np.clip(row_delta / temperature, 0.0, 60.0))
    affinity[~valid] = 0.0

    degree = valid.sum(axis=1, keepdims=True).astype(float)
    floor_mass = 0.035
    floor = np.divide(
        floor_mass * valid.astype(float),
        degree,
        out=np.zeros_like(d),
        where=degree > 0.0,
    )

    local = affinity + floor
    row_sum = local.sum(axis=1, keepdims=True)
    local = np.divide(
        local,
        row_sum,
        out=np.zeros_like(local),
        where=row_sum > 0.0,
    )

    balanced = local.copy()
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

    global_local = 0.72 * balanced + 0.28 * local

    row_regret = np.exp(-row_delta / row_gap_scale)
    col_regret = np.exp(-col_delta / col_gap_scale)
    regret = np.sqrt(row_regret * col_regret)

    nearest_row = valid & (d <= row_min[:, None] + tiny)
    nearest_col = valid & (d <= col_min[None, :] + tiny)
    mutual = nearest_row & nearest_col

    correction = np.power(regret + tiny, 0.8)
    correction *= 1.0 + 0.10 * nearest_row
    correction *= 1.0 + 0.16 * mutual

    score = global_local * correction
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
