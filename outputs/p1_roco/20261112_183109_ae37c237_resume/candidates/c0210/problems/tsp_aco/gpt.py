import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, row-normalized edge-priority indicators for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.nan)

    global_values = d[valid]
    positive_values = global_values[global_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(global_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    row_counts = valid.sum(axis=1)
    col_counts = valid.sum(axis=0)

    row_medians = np.nanmedian(work, axis=1)
    col_medians = np.nanmedian(work, axis=0)

    row_medians = np.where(
        np.isfinite(row_medians) & (row_medians > 0.0),
        row_medians,
        global_scale,
    )
    col_medians = np.where(
        np.isfinite(col_medians) & (col_medians > 0.0),
        col_medians,
        global_scale,
    )

    row_medians = np.maximum(row_medians, eps)
    col_medians = np.maximum(col_medians, eps)

    row_relative = np.divide(
        d,
        row_medians[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    col_relative = np.divide(
        d,
        col_medians[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    row_deviation = np.abs(np.log1p(row_relative))
    col_deviation = np.abs(np.log1p(col_relative))

    local_cost = 0.5 * (row_deviation + col_deviation)

    base = np.divide(
        1.0,
        1.0 + local_cost,
        out=np.zeros_like(d),
        where=valid,
    )

    zero_bonus = np.where(valid & (d == 0.0), 1.15, 1.0)

    row_accessibility = np.sum(
        np.where(valid, 1.0 / (1.0 + row_relative), 0.0),
        axis=1,
    )
    col_accessibility = np.sum(
        np.where(valid, 1.0 / (1.0 + col_relative), 0.0),
        axis=0,
    )

    row_density = np.divide(
        row_accessibility,
        np.maximum(row_counts, 1),
        out=np.zeros(n, dtype=float),
        where=row_counts > 0,
    )
    col_density = np.divide(
        col_accessibility,
        np.maximum(col_counts, 1),
        out=np.zeros(n, dtype=float),
        where=col_counts > 0,
    )

    row_scarcity = 1.0 / (1.0 + row_density)
    col_scarcity = 1.0 / (1.0 + col_density)

    score = (
        base
        * zero_bonus
        * (0.75 + 0.25 * row_scarcity[:, None])
        * (0.75 + 0.25 * col_scarcity[None, :])
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    for _ in range(12):
        row_totals = score.sum(axis=1, keepdims=True)
        row_scale = np.divide(
            1.0,
            row_totals,
            out=np.zeros_like(row_totals),
            where=row_totals > 0.0,
        )
        score *= row_scale

        col_totals = score.sum(axis=0, keepdims=True)
        col_scale = np.divide(
            1.0,
            col_totals,
            out=np.zeros_like(col_totals),
            where=col_totals > 0.0,
        )
        score *= col_scale

        score[~valid] = 0.0

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
