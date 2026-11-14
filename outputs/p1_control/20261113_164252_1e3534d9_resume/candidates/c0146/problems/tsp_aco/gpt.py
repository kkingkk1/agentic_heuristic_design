import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return exchangeability-aware stochastic edge priorities for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)

    positive = work[valid & (work > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite = work[valid]
        global_scale = float(np.median(finite)) if finite.size else 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    row_values = np.where(valid, work, np.nan)
    row_median = np.nanmedian(row_values, axis=1)
    col_median = np.nanmedian(row_values, axis=0)

    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0.0),
        row_median,
        global_scale,
    )
    col_median = np.where(
        np.isfinite(col_median) & (col_median > 0.0),
        col_median,
        global_scale,
    )

    endpoint_scale = np.sqrt(
        row_median[:, None] * col_median[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    normalized_cost = np.zeros_like(d, dtype=float)
    np.divide(
        work,
        endpoint_scale,
        out=normalized_cost,
        where=valid,
    )

    # Cheap edges relative to the typical scale of both endpoints.
    cost_signal = 1.0 / (1.0 + normalized_cost)

    # Find the best valid two-hop replacement for every edge.
    best_detour = np.full((n, n), np.inf, dtype=float)
    near_detours = np.zeros((n, n), dtype=float)

    detour_tolerance = 0.12 * global_scale

    for k in range(n):
        left = work[:, k][:, None]
        right = work[k, :][None, :]
        detour_valid = np.isfinite(left) & np.isfinite(right)

        detour = left + right
        detour[~detour_valid] = np.inf

        best_detour = np.minimum(best_detour, detour)

        near = detour <= work + detour_tolerance
        near_detours += near.astype(float)

    possible_intermediates = max(n - 2, 1)
    replacement_fraction = near_detours / float(possible_intermediates)

    finite_detour = np.isfinite(best_detour) & valid
    shortcut_gap = np.zeros_like(d, dtype=float)
    shortcut_gap[finite_detour] = np.maximum(
        best_detour[finite_detour] - work[finite_detour],
        0.0,
    )

    gap_scale_values = shortcut_gap[finite_detour & (shortcut_gap > 0.0)]
    gap_scale = (
        float(np.median(gap_scale_values))
        if gap_scale_values.size
        else global_scale
    )
    gap_scale = max(gap_scale, eps)

    # Reward structurally distinctive direct edges.
    bridge_signal = np.tanh(shortcut_gap / gap_scale)

    # Penalize edges with many nearly equivalent two-hop replacements.
    exclusivity = 1.0 / (
        1.0 + 1.8 * replacement_fraction
    )

    score = cost_signal * (
        0.58
        + 0.27 * bridge_signal
        + 0.15 * exclusivity
    )

    # Mild mass correction discourages concentrating all probability
    # on a small number of rows or columns.
    score[~valid] = 0.0
    row_mass = np.sum(score, axis=1)
    col_mass = np.sum(score, axis=0)

    positive_row = row_mass[row_mass > 0.0]
    positive_col = col_mass[col_mass > 0.0]

    row_reference = (
        float(np.median(positive_row))
        if positive_row.size
        else 1.0
    )
    col_reference = (
        float(np.median(positive_col))
        if positive_col.size
        else 1.0
    )

    row_factor = np.sqrt(
        row_reference / np.maximum(row_mass, eps)
    )
    col_factor = np.sqrt(
        col_reference / np.maximum(col_mass, eps)
    )

    score *= row_factor[:, None] * col_factor[None, :]
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal_valid = valid & ~np.eye(n, dtype=bool)
    mean_score = float(np.mean(score[off_diagonal_valid]))

    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
