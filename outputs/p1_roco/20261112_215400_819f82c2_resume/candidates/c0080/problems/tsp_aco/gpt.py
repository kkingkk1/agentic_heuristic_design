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
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = ~np.eye(n, dtype=bool)
    raw = np.maximum(distance_matrix, 0.0)
    np.fill_diagonal(raw, 0.0)

    symmetric = 0.5 * (raw + raw.T)
    off_raw = raw[valid]

    if off_raw.size == 0:
        return result

    asymmetry = np.abs(raw - raw.T)[valid]
    asym_scale = float(np.median(np.abs(off_raw))) + np.finfo(float).eps
    asym_ratio = float(np.median(asymmetry)) / asym_scale
    directional_weight = float(np.clip(asym_ratio, 0.0, 0.35))

    effective = (
        (1.0 - directional_weight) * symmetric
        + directional_weight * raw
    )
    np.fill_diagonal(effective, 0.0)

    positive = effective[valid]
    positive = positive[positive > 0.0]

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

    def robust_scale(values: np.ndarray) -> float:
        if values.size == 0:
            return global_scale

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = median + 1.4826 * mad

        if scale <= floor:
            positive_values = values[values > 0.0]
            if positive_values.size:
                scale = float(np.median(positive_values))
            else:
                scale = global_scale

        return max(scale, floor)

    row_scales = np.empty(n, dtype=float)
    col_scales = np.empty(n, dtype=float)

    for i in range(n):
        row_scales[i] = robust_scale(effective[i, np.arange(n) != i])
        col_scales[i] = robust_scale(effective[np.arange(n) != i, i])

    pair_scale = np.sqrt(
        np.maximum(row_scales[:, None], floor)
        * np.maximum(col_scales[None, :], floor)
    )
    pair_scale = np.maximum(pair_scale, floor)

    locality = np.zeros((n, n), dtype=float)
    locality[valid] = np.exp(-effective[valid] / pair_scale[valid])

    ranks = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        values = effective[i, indices]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]

        rank_values = np.empty(sorted_values.size, dtype=float)
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

        row_ranks = np.empty_like(rank_values)
        row_ranks[order] = rank_values

        denominator = max(sorted_values.size - 1, 1)
        ranks[i, indices] = row_ranks / denominator

    scarcity = np.clip(1.0 - ranks, 0.0, 1.0)

    nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = effective[i].copy()
        row[i] = np.inf
        minimum = float(np.min(row))

        if np.isfinite(minimum):
            nearest[i] = (row == minimum).astype(float)
            nearest[i, i] = 0.0

    mutual_nearest = nearest * nearest.T
    mutual_scarcity = np.sqrt(scarcity * scarcity.T)

    zero_bonus = ((effective == 0.0) & valid).astype(float)
    zero_bonus *= (locality >= np.max(locality[valid]) - 1e-15).astype(float)

    score = (
        0.66 * locality
        + 0.24 * mutual_scarcity
        + 0.08 * mutual_nearest
        + 0.02 * zero_bonus
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
