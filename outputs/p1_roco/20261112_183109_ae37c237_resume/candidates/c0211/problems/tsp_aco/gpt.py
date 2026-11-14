import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-contrast edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    row_medians = np.empty(n, dtype=float)
    col_medians = np.empty(n, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        row_medians[i] = np.median(values) if values.size else 1.0

    for j in range(n):
        values = d[valid[:, j], j]
        col_medians[j] = np.median(values) if values.size else 1.0

    finite_values = d[valid]
    global_scale = np.median(finite_values) if finite_values.size else 1.0
    global_scale = max(float(global_scale), eps)

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

    row_order = np.argsort(work, axis=1, kind="stable")
    col_order = np.argsort(work, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    valid_row_counts = valid.sum(axis=1).astype(float)
    valid_col_counts = valid.sum(axis=0).astype(float)

    row_fraction = np.divide(
        row_rank,
        np.maximum(valid_row_counts[:, None] - 1.0, 1.0),
    )
    col_fraction = np.divide(
        col_rank,
        np.maximum(valid_col_counts[None, :] - 1.0, 1.0),
    )

    rank_score = np.sqrt(
        np.maximum(0.0, 1.0 - row_fraction)
        * np.maximum(0.0, 1.0 - col_fraction)
    )

    row_contrast = np.divide(
        row_medians[:, None] - d,
        row_medians[:, None] + d + eps,
        out=np.zeros_like(d),
        where=valid,
    )
    col_contrast = np.divide(
        col_medians[None, :] - d,
        col_medians[None, :] + d + eps,
        out=np.zeros_like(d),
        where=valid,
    )

    contrast_score = 0.5 + 0.5 * np.tanh(
        2.5 * np.minimum(row_contrast, col_contrast)
    )

    reciprocal_score = np.divide(
        1.0,
        1.0 + d / global_scale,
        out=np.zeros_like(d),
        where=valid,
    )

    score = (
        0.50 * rank_score
        + 0.30 * contrast_score
        + 0.20 * reciprocal_score
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
