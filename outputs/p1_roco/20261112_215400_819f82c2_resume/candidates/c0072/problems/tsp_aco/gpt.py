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
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(distance_matrix, 0.0).copy()
    valid = ~np.eye(n, dtype=bool)
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

    def robust_scales(values: np.ndarray) -> np.ndarray:
        scales = np.empty(values.shape[0], dtype=float)

        for i in range(values.shape[0]):
            row = values[i]
            row = row[np.isfinite(row)]

            if row.size == 0:
                scales[i] = global_scale
                continue

            median = float(np.median(row))
            mad = float(np.median(np.abs(row - median)))
            scale = median + 1.4826 * mad

            if scale <= floor:
                positive_row = row[row > 0.0]
                if positive_row.size:
                    scale = float(np.median(positive_row))

            scales[i] = max(scale, floor)

        return scales

    work = d.copy()
    np.fill_diagonal(work, np.nan)

    row_scale = robust_scales(work)
    col_scale = robust_scales(work.T)

    row_affinity = np.zeros_like(d)
    col_affinity = np.zeros_like(d)

    row_affinity[valid] = np.exp(
        -d[valid] / row_scale[np.nonzero(valid)[0]]
    )
    col_affinity[valid] = np.exp(
        -d[valid] / col_scale[np.nonzero(valid)[1]]
    )

    def tie_aware_ranks(values: np.ndarray) -> np.ndarray:
        ranks = np.zeros_like(values, dtype=float)

        for i in range(values.shape[0]):
            row = values[i]
            finite = np.isfinite(row)
            finite_values = row[finite]

            if finite_values.size == 0:
                continue

            ranks[i, finite] = np.sum(
                finite_values[None, :] < finite_values[:, None], axis=1
            )
            equal_counts = np.sum(
                finite_values[None, :] == finite_values[:, None], axis=1
            )
            ranks[i, finite] += 0.5 * (equal_counts - 1.0)

        return ranks

    row_rank = tie_aware_ranks(work)
    col_rank = tie_aware_ranks(work.T)

    denom = max(float(n - 2), 1.0)
    row_scarcity = np.clip(1.0 - row_rank / denom, 0.0, 1.0)
    col_scarcity = np.clip(1.0 - col_rank / denom, 0.0, 1.0)

    mutual_bonus = np.sqrt(row_scarcity * col_scarcity)
    zero_bonus = (d == 0.0).astype(float)
    zero_bonus[~valid] = 0.0

    score = (
        0.48 * row_affinity
        + 0.22 * col_affinity
        + 0.18 * mutual_bonus
        + 0.12 * zero_bonus
    )

    score[~valid] = 0.0

    values = score[valid]
    low = float(np.min(values))
    high = float(np.max(values))

    if high > low:
        score[valid] = (values - low) / (high - low)
    elif high > 0.0:
        score[valid] = 1.0
    else:
        score[valid] = 0.0

    np.fill_diagonal(score, 0.0)
    return np.clip(score, 0.0, 1.0)
