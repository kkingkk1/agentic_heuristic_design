import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, detour-aware edge-promise matrix for TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    reverse_valid = valid.T
    both_valid = valid & reverse_valid
    one_way = valid | reverse_valid

    symmetric = np.zeros_like(raw, dtype=float)
    symmetric[both_valid] = 0.5 * (
        raw[both_valid] + raw.T[both_valid]
    )
    symmetric[valid & ~reverse_valid] = raw[valid & ~reverse_valid]
    symmetric[~valid & reverse_valid] = raw.T[~valid & reverse_valid]

    finite_edges = symmetric[one_way]
    positive_edges = finite_edges[finite_edges > 0.0]
    base_scale = (
        float(np.median(positive_edges))
        if positive_edges.size
        else max(float(np.median(finite_edges)), 1.0)
    )
    eps = np.finfo(float).eps
    base_scale = max(base_scale, eps)

    masked = np.where(one_way, symmetric, np.inf)

    row_median = np.median(
        np.where(np.isfinite(masked), masked, np.nan),
        axis=1,
    )
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0.0),
        row_median,
        base_scale,
    )
    row_median = np.maximum(row_median, eps)

    pair_scale = np.sqrt(
        row_median[:, None] * row_median[None, :]
    )

    normalized = np.zeros_like(symmetric)
    np.divide(
        symmetric,
        pair_scale,
        out=normalized,
        where=one_way,
    )

    positive_normalized = normalized[one_way & (normalized > 0.0)]
    norm_scale = (
        float(np.median(positive_normalized))
        if positive_normalized.size
        else 1.0
    )
    norm_scale = max(norm_scale, eps)

    centered_length = normalized / norm_scale
    shortness = 1.0 / (1.0 + np.square(np.maximum(centered_length, 0.0)))

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_excess = np.maximum(
        symmetric - row_best[:, None],
        0.0,
    )
    col_excess = np.maximum(
        symmetric - col_best[None, :],
        0.0,
    )

    row_gap = np.partition(masked, min(1, n - 1), axis=1)[:, min(1, n - 1)]
    col_gap = np.partition(masked, min(1, n - 1), axis=0)[min(1, n - 1), :]

    row_gap = np.maximum(row_gap - row_best, 0.0)
    col_gap = np.maximum(col_gap - col_best, 0.0)

    finite_row_gap = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    finite_col_gap = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(finite_row_gap))
        if finite_row_gap.size
        else base_scale
    )
    col_gap_scale = (
        float(np.median(finite_col_gap))
        if finite_col_gap.size
        else base_scale
    )
    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    endpoint_need = 0.5 * (
        1.0 / (1.0 + row_excess / row_gap_scale)
        + 1.0 / (1.0 + col_excess / col_gap_scale)
    )

    detour = np.full((n, n), np.inf, dtype=float)
    for middle in range(n):
        candidate = masked[:, middle, None] + masked[None, middle, :]
        detour = np.minimum(detour, candidate)

    detour_ratio = np.zeros_like(symmetric)
    usable_detour = one_way & np.isfinite(detour)
    np.divide(
        symmetric,
        detour,
        out=detour_ratio,
        where=usable_detour,
    )

    detour_advantage = np.zeros_like(symmetric)
    detour_advantage[usable_detour] = 1.0 / (
        1.0 + np.square(detour_ratio[usable_detour])
    )

    degree_load = np.zeros(n, dtype=float)
    degree_load[:] = np.sum(
        np.where(one_way, 1.0 / (1.0 + symmetric / base_scale), 0.0),
        axis=1,
    )
    load_scale = max(float(np.median(degree_load)), eps)
    endpoint_balance = 1.0 / (
        1.0
        + 0.5
        * (
            degree_load[:, None] / load_scale
            + degree_load[None, :] / load_scale
        )
    )

    score = (
        0.42 * shortness
        + 0.30 * endpoint_need
        + 0.20 * detour_advantage
        + 0.08 * endpoint_balance
    )

    score[~one_way] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
