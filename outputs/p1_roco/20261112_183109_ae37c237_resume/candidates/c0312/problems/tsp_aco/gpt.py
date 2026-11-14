import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized stochastic priorities for TSP edges."""
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
    safe_d = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    positive_rows = []
    for i in range(n):
        values = d[i, valid[i] & (d[i] > 0.0)]
        positive_rows.append(
            float(np.median(values)) if values.size else global_scale
        )
    row_scale = np.maximum(np.asarray(positive_rows), eps)

    positive_cols = []
    for j in range(n):
        values = d[valid[:, j] & (d[:, j] > 0.0), j]
        positive_cols.append(
            float(np.median(values)) if values.size else global_scale
        )
    col_scale = np.maximum(np.asarray(positive_cols), eps)

    local_ratio_row = np.divide(
        d,
        row_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    local_ratio_col = np.divide(
        d,
        col_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_affinity = np.exp(-np.clip(local_ratio_row, 0.0, 50.0))
    col_affinity = np.exp(-np.clip(local_ratio_col, 0.0, 50.0))

    row_sum = row_affinity.sum(axis=1, keepdims=True)
    col_sum = col_affinity.sum(axis=0, keepdims=True)

    row_prob = np.divide(
        row_affinity,
        row_sum,
        out=np.zeros_like(d),
        where=row_sum > 0.0,
    )
    col_prob = np.divide(
        col_affinity,
        col_sum,
        out=np.zeros_like(d),
        where=col_sum > 0.0,
    )

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_entropy = -np.sum(
        np.where(row_prob > 0.0, row_prob * np.log(row_prob), 0.0),
        axis=1,
    )
    col_entropy = -np.sum(
        np.where(col_prob > 0.0, col_prob * np.log(col_prob), 0.0),
        axis=0,
    )

    row_entropy = np.divide(
        row_entropy,
        np.log(np.maximum(row_count, 2)),
        out=np.zeros(n, dtype=float),
        where=row_count > 1,
    )
    col_entropy = np.divide(
        col_entropy,
        np.log(np.maximum(col_count, 2)),
        out=np.zeros(n, dtype=float),
        where=col_count > 1,
    )

    row_entropy = np.clip(row_entropy, 0.0, 1.0)
    col_entropy = np.clip(col_entropy, 0.0, 1.0)

    global_ratio = np.divide(
        d,
        global_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    global_quality = np.exp(-np.clip(global_ratio, 0.0, 50.0) / 1.6)

    row_global_mix = 0.08 + 0.16 * row_entropy
    col_global_mix = 0.08 + 0.16 * col_entropy

    row_quality = (
        (1.0 - row_global_mix[:, None]) * row_affinity
        + row_global_mix[:, None] * global_quality
    )
    col_quality = (
        (1.0 - col_global_mix[None, :]) * col_affinity
        + col_global_mix[None, :] * global_quality
    )

    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(safe_d, axis=1)
    row_ranks = np.empty((n, n), dtype=float)
    row_ranks[np.arange(n)[:, None], row_order] = (
        np.arange(n, dtype=float)[None, :] + 1.0
    )

    col_order = np.argsort(safe_d, axis=0)
    col_ranks = np.empty((n, n), dtype=float)
    col_ranks[col_order, np.arange(n)[None, :]] = (
        np.arange(n, dtype=float)[:, None] + 1.0
    )

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.06 * np.maximum(row_ranks - 1.0, 0.0))
        * (1.0 + 0.06 * np.maximum(col_ranks - 1.0, 0.0))
    )

    row_support = np.maximum(row_affinity.sum(axis=1), eps)
    col_support = np.maximum(col_affinity.sum(axis=0), eps)

    normalized_row_support = np.sqrt(
        row_support / np.maximum(row_count, 1)
    )
    normalized_col_support = np.sqrt(
        col_support / np.maximum(col_count, 1)
    )

    bridge = 1.0 + 0.06 / np.sqrt(
        np.maximum(normalized_row_support[:, None], eps)
        * np.maximum(normalized_col_support[None, :], eps)
    )

    edge_mass_row = row_affinity
    edge_mass_col = col_affinity

    residual_row = np.maximum(
        row_sum - edge_mass_row,
        0.0,
    )
    residual_col = np.maximum(
        col_sum - edge_mass_col,
        0.0,
    )

    residual_row_fraction = np.divide(
        residual_row,
        np.maximum(row_sum, eps),
        out=np.zeros_like(d),
        where=row_sum > 0.0,
    )
    residual_col_fraction = np.divide(
        residual_col,
        np.maximum(col_sum, eps),
        out=np.zeros_like(d),
        where=col_sum > 0.0,
    )

    residual_support = np.sqrt(
        residual_row_fraction * residual_col_fraction
    )
    future_factor = 1.0 + 0.08 * (1.0 - residual_support)

    score = (
        mutual_quality
        * rank_quality
        * bridge
        * future_factor
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    floor = 0.02 * row_max
    score = np.where(valid, np.maximum(score, floor), 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
