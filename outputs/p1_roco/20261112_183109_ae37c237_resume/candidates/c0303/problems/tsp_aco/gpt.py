import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diversity-aware row-normalized edge priorities for stochastic TSP sampling."""
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

    positive_values = d[valid]
    global_scale = float(np.median(positive_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        positive = positive_values[positive_values > 0.0]
        global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(4, n - 1)
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)

    safe_nearest = np.where(finite_nearest, nearest, 0.0)
    local_scale = np.divide(
        safe_nearest.sum(axis=1),
        np.maximum(counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    local_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    global_distance = d / global_scale

    local_quality = np.exp(-np.clip(local_distance, 0.0, 50.0) / 1.05)
    global_quality = np.exp(-np.clip(global_distance, 0.0, 50.0) / 1.35)

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    endpoint_compatibility = np.sqrt(
        np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.10)
        * np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.10)
    )

    row_order = np.argsort(np.argsort(work, axis=1), axis=1)
    col_order = np.argsort(np.argsort(work, axis=0), axis=0)

    row_rank = np.where(valid, row_order + 1.0, 0.0)
    col_rank = np.where(valid, col_order + 1.0, 0.0)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.075 * row_rank) * (1.0 + 0.075 * col_rank)
    )

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.10 * regret

    neighborhood = np.zeros_like(d, dtype=float)
    for i in range(n):
        row_valid = np.flatnonzero(valid[i])
        if row_valid.size:
            chosen = row_valid[np.argsort(d[i, row_valid])[:k]]
            neighborhood[i, chosen] = 1.0

    neighborhood_counts = neighborhood.sum(axis=1)
    overlap = neighborhood @ neighborhood.T
    denominator = np.sqrt(
        np.maximum(neighborhood_counts[:, None], 1.0)
        * np.maximum(neighborhood_counts[None, :], 1.0)
    )
    overlap = np.divide(
        overlap,
        denominator,
        out=np.zeros_like(overlap),
        where=denominator > 0.0,
    )
    overlap = 0.5 * (overlap + overlap.T)

    bridge_potential = 1.0 + 0.18 * (1.0 - overlap)
    dense_cluster_penalty = np.exp(-0.24 * overlap)

    core = (
        local_quality
        * endpoint_compatibility
        * rank_quality
        * bridge_potential
        * dense_cluster_penalty
        * urgency[:, None]
        * urgency[None, :]
    )

    core[~valid] = 0.0

    local_row_sums = local_quality.sum(axis=1, keepdims=True)
    local_prob = np.divide(
        local_quality,
        local_row_sums,
        out=np.zeros_like(local_quality),
        where=local_row_sums > 0.0,
    )

    entropy = -np.sum(
        np.where(local_prob > 0.0, local_prob * np.log(local_prob), 0.0),
        axis=1,
    )
    entropy_scale = np.log(max(k, 2))
    concentration = np.clip(1.0 - entropy / entropy_scale, 0.0, 1.0)

    global_row_sums = global_quality.sum(axis=1, keepdims=True)
    global_prob = np.divide(
        global_quality,
        global_row_sums,
        out=np.zeros_like(global_quality),
        where=global_row_sums > 0.0,
    )

    global_mix = 0.14 + 0.18 * concentration
    score = (
        (1.0 - global_mix[:, None]) * core
        + global_mix[:, None] * global_prob
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
