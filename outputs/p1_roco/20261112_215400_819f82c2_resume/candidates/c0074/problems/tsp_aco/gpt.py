import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = ~np.eye(n, dtype=bool)

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diag = d[valid]
    positive = off_diag[off_diag > 0.0]

    if positive.size:
        global_median = float(np.median(positive))
        global_mad = float(np.median(np.abs(positive - global_median)))
        global_scale = global_median + 1.4826 * global_mad
        if global_scale <= 0.0:
            global_scale = global_median
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    floor = max(global_scale * 1e-3, eps)

    masked = d.copy()
    masked[~valid] = np.nan

    row_scales = np.empty(n, dtype=float)
    for i in range(n):
        values = masked[i, np.isfinite(masked[i])]
        if values.size == 0:
            row_scales[i] = global_scale
            continue

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = median + 1.4826 * mad

        if scale <= floor:
            positive_values = values[values > 0.0]
            scale = (
                float(np.median(positive_values))
                if positive_values.size
                else global_scale
            )

        row_scales[i] = max(scale, floor)

    scale = 0.5 * (row_scales[:, None] + row_scales[None, :])
    scale = np.maximum(scale, floor)

    locality = np.zeros((n, n), dtype=float)
    locality[valid] = np.exp(-d[valid] / scale[valid])

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        values = d[i, valid[i]]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]

        rank_values = np.empty_like(sorted_values, dtype=float)
        start = 0
        while start < sorted_values.size:
            end = start + 1
            while (
                end < sorted_values.size
                and sorted_values[end] == sorted_values[start]
            ):
                end += 1
            rank_values[start:end] = 0.5 * (start + end - 1)
            start = end

        row_rank = np.empty_like(rank_values)
        row_rank[order] = rank_values
        ranks[i, valid[i]] = row_rank

    rank_denominator = max(float(n - 2), 1.0)
    scarcity = np.clip(1.0 - ranks / rank_denominator, 0.0, 1.0)
    mutual_scarcity = np.sqrt(scarcity * scarcity.T)

    nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        minimum = np.min(row)
        if np.isfinite(minimum):
            nearest[i] = np.isclose(row, minimum, rtol=0.0, atol=0.0).astype(float)
            nearest[i, i] = 0.0

    mutual_nearest = nearest * nearest.T
    zero_bonus = ((d == 0.0) & valid).astype(float)

    score = (
        0.56 * locality
        + 0.24 * mutual_scarcity
        + 0.12 * mutual_nearest
        + 0.08 * zero_bonus
    )
    score[~valid] = 0.0

    values = score[valid]
    low = float(np.min(values))
    high = float(np.max(values))

    if high > low:
        score[valid] = (values - low) / (high - low)
    elif high > 0.0:
        score[valid] = 1.0

    np.fill_diagonal(score, 0.0)
    return np.clip(score, 0.0, 1.0)
