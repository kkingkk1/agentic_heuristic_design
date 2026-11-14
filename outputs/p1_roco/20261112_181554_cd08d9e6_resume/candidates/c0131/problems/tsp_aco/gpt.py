import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, globally diverse TSP edge-priority indicators."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    # Use the union of directed valid edges so asymmetric input does not
    # silently discard potentially useful connections.
    valid = finite_nonnegative | finite_nonnegative.T
    if not np.any(valid):
        return result

    sym_d = np.zeros_like(d, dtype=float)
    both = finite_nonnegative & finite_nonnegative.T
    only_forward = finite_nonnegative & ~finite_nonnegative.T
    only_backward = finite_nonnegative.T & ~finite_nonnegative

    sym_d[both] = 0.5 * (d[both] + d.T[both])
    sym_d[only_forward] = d[only_forward]
    sym_d[only_backward] = d.T[only_backward]

    positive_values = sym_d[valid & (sym_d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    inf = np.inf
    work = np.where(valid, sym_d, inf)
    sorted_distances = np.sort(work, axis=1)

    k = min(max(3, int(np.sqrt(n))), n - 1)
    local_scale = np.full(n, global_scale, dtype=float)
    best = np.full(n, inf, dtype=float)
    second = np.full(n, inf, dtype=float)
    quantile = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = sorted_distances[i]
        finite_row = row[np.isfinite(row)]
        if finite_row.size:
            best[i] = finite_row[0]
            if finite_row.size > 1:
                second[i] = finite_row[1]
            local_values = finite_row[: min(k, finite_row.size)]
            local_scale[i] = max(
                float(np.median(local_values)),
                np.finfo(float).eps,
            )
            quantile[i] = float(
                finite_row[min(finite_row.size - 1, max(1, k // 2))]
            )

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        np.maximum(safe_best, local_scale),
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    row_ratio = np.divide(
        sym_d,
        local_scale[:, None],
        out=np.full_like(sym_d, inf),
        where=valid,
    )
    col_ratio = np.divide(
        sym_d,
        local_scale[None, :],
        out=np.full_like(sym_d, inf),
        where=valid,
    )

    local_quality = np.exp(-np.clip(row_ratio + col_ratio, 0.0, 60.0) * 0.5)
    absolute_quality = np.exp(-np.clip(sym_d / global_scale, 0.0, 60.0))

    row_order = np.argsort(row_ratio, axis=1, kind="stable")
    row_rank = np.empty_like(row_order)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]

    col_order = np.argsort(col_ratio, axis=0, kind="stable")
    col_rank = np.empty_like(col_order)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    rank_quality = 1.0 / np.sqrt(
        1.0 + row_rank.astype(float) + col_rank.astype(float)
    )

    # Endpoint scarcity: edges are valuable when they preserve access to
    # endpoints with few competitive alternatives.
    reserve = np.minimum(
        safe_second / np.maximum(quantile, np.finfo(float).eps),
        6.0,
    )
    reserve_factor = 1.0 + 0.18 * np.tanh(
        0.5 * (reserve[:, None] + reserve[None, :])
    )

    urgency = np.sqrt(
        (1.0 + 0.85 * regret[:, None])
        * (1.0 + 0.85 * regret[None, :])
    )

    # Compact neighborhoods define local basins.
    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(sym_d[i, candidates], kind="stable")]
            neighborhood[i, order[:k]] = 1.0

    degree = neighborhood.sum(axis=1)
    degree = np.maximum(degree, 1.0)

    overlap = (neighborhood @ neighborhood.T) / np.minimum(
        degree[:, None],
        degree[None, :],
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    # Bridge strength rewards edges whose endpoints belong to different
    # neighborhoods, but keeps a nonzero preference for local edges.
    bridge_factor = 1.0 + 0.32 * (1.0 - overlap)

    # Two-hop access estimates whether an edge helps connect otherwise
    # poorly overlapping portions of the candidate graph.
    reach = neighborhood @ neighborhood.T
    reach_scale = np.maximum(reach.max(axis=1), 1.0)
    access_i = np.divide(
        reach.sum(axis=1),
        reach_scale,
        out=np.zeros(n, dtype=float),
        where=reach_scale > 0.0,
    )
    access_factor = 1.0 + 0.12 * (
        1.0 - np.sqrt(access_i[:, None] * access_i[None, :])
    )

    # Mutual-nearest edges receive a restrained bonus rather than dominating
    # the global exploration signal.
    nearest = valid & (
        sym_d <= safe_best[:, None]
        + 1e-12 * max(global_scale, 1.0)
    )
    mutual_nearest = nearest & nearest.T
    nearest_factor = 1.0 + 0.16 * mutual_nearest.astype(float)

    score = (
        local_quality
        * absolute_quality
        * rank_quality
        * urgency
        * reserve_factor
        * bridge_factor
        * access_factor
        * nearest_factor
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    total = float(score.sum())
    if not np.isfinite(total) or total <= 0.0:
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / count
        return result

    normalized = score / total

    # Adaptive exploration floor: concentrated scores receive a stronger
    # floor, while already diverse scores retain their learned contrast.
    positive = normalized[valid]
    concentration = float(np.sum(positive * positive))
    valid_count = float(positive.size)
    concentration_ratio = np.clip(concentration * valid_count, 1.0, valid_count)
    floor_strength = 0.018 + 0.022 * (
        (concentration_ratio - 1.0) / max(valid_count - 1.0, 1.0)
    )
    floor = floor_strength / valid_count

    result = (1.0 - floor_strength) * normalized
    result[valid] += floor
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return result
