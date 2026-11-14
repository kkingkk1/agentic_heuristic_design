import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive stochastic priorities for TSP edge sampling."""
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
    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(valid_values))

    if not np.isfinite(global_scale) or global_scale <= eps:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(5, n - 1)
    nearest = sorted_rows[:, :k]
    nearest_finite = np.isfinite(nearest)
    nearest_sum = np.where(nearest_finite, nearest, 0.0).sum(axis=1)
    nearest_count = nearest_finite.sum(axis=1)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
    )
    local_scale = np.where(local_scale > eps, local_scale, global_scale)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    pair_scale = np.sqrt(row_scale * col_scale)

    local_ratio = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    global_ratio = np.divide(
        d,
        global_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    local_quality = np.exp(-np.clip(local_ratio, 0.0, 50.0) / 1.05)
    global_quality = np.exp(-np.clip(global_ratio, 0.0, 50.0) / 1.55)

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

    mutual_quality = np.sqrt(
        np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.10)
        * np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.10)
    )

    row_order = np.argsort(work, axis=1)
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = (
        np.arange(n, dtype=float)[None, :] + 1.0
    )

    col_order = np.argsort(work, axis=0)
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = (
        np.arange(n, dtype=float)[:, None] + 1.0
    )

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.07 * np.maximum(row_rank - 1.0, 0.0))
        * (1.0 + 0.07 * np.maximum(col_rank - 1.0, 0.0))
    )

    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = np.argsort(d[i, candidates])
            neighborhood[i, candidates[order[:k]]] = 1.0

    neighborhood_counts = neighborhood.sum(axis=1)
    overlap = neighborhood @ neighborhood.T
    overlap_denominator = np.sqrt(
        np.maximum(neighborhood_counts[:, None], 1.0)
        * np.maximum(neighborhood_counts[None, :], 1.0)
    )
    overlap = np.divide(
        overlap,
        overlap_denominator,
        out=np.zeros_like(overlap),
        where=overlap_denominator > 0.0,
    )
    overlap = 0.5 * (overlap + overlap.T)
    np.fill_diagonal(overlap, 1.0)

    bridge_potential = 1.0 + 0.14 * (1.0 - overlap)
    cluster_penalty = np.exp(-0.10 * overlap)

    local_sums = local_quality.sum(axis=1, keepdims=True)
    local_prob = np.divide(
        local_quality,
        local_sums,
        out=np.zeros_like(local_quality),
        where=local_sums > 0.0,
    )

    entropy = -np.sum(
        np.where(local_prob > 0.0, local_prob * np.log(local_prob), 0.0),
        axis=1,
    )
    entropy_reference = np.log(max(2, min(k, n - 1)))
    concentration = np.clip(
        1.0 - entropy / max(entropy_reference, eps),
        0.0,
        1.0,
    )

    global_sums = global_quality.sum(axis=1, keepdims=True)
    global_prob = np.divide(
        global_quality,
        global_sums,
        out=np.zeros_like(global_quality),
        where=global_sums > 0.0,
    )
    global_mix = np.clip(0.10 + 0.16 * concentration, 0.10, 0.26)

    # Edge-specific future potential after reserving this edge.
    continuation = local_quality.copy()
    np.fill_diagonal(continuation, 0.0)

    outgoing_total = continuation.sum(axis=1, keepdims=True)
    incoming_total = continuation.sum(axis=0, keepdims=True)

    outgoing_remaining = np.maximum(outgoing_total - continuation, 0.0)
    incoming_remaining = np.maximum(incoming_total - continuation, 0.0)

    outgoing_fraction = np.divide(
        outgoing_remaining,
        outgoing_total,
        out=np.zeros_like(continuation),
        where=outgoing_total > 0.0,
    )
    incoming_fraction = np.divide(
        incoming_remaining,
        incoming_total,
        out=np.zeros_like(continuation),
        where=incoming_total > 0.0,
    )

    future_potential = 0.86 + 0.24 * np.sqrt(
        np.clip(outgoing_fraction * incoming_fraction, 0.0, 1.0)
    )

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    scarcity = np.clip(
        (safe_second - safe_best)
        / np.maximum(np.abs(safe_second), eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.09 * scarcity

    score = (
        ((1.0 - global_mix)[:, None] * local_quality
         + global_mix[:, None] * global_quality)
        * mutual_quality
        * rank_quality
        * bridge_potential
        * cluster_penalty
        * future_potential
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    floor = 0.015 * row_max
    score = np.where(valid, np.maximum(score, floor), 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
