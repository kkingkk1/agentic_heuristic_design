import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diversity-aware, long-term edge priorities for stochastic TSP sampling."""
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

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.median(valid_values))
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)

    max_k = max(1, n - 1)
    k = min(5, max_k)

    sorted_rows = np.sort(work, axis=1)
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
    local_scale = np.maximum(local_scale, eps)

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
    global_quality = np.exp(-np.clip(global_ratio, 0.0, 50.0) / 1.45)

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
        np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.15)
        * np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.15)
    )

    row_order = np.argsort(np.argsort(work, axis=1), axis=1)
    col_order = np.argsort(np.argsort(work, axis=0), axis=0)

    row_rank = np.where(valid, row_order + 1.0, 0.0)
    col_rank = np.where(valid, col_order + 1.0, 0.0)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.065 * row_rank) * (1.0 + 0.065 * col_rank)
    )

    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = np.argsort(d[i, candidates])
            chosen = candidates[order[:k]]
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
    np.fill_diagonal(overlap, 1.0)

    # Mild bridge preference: connect regions with different local neighborhoods.
    bridge_potential = 1.0 + 0.16 * (1.0 - overlap)

    # Keep the cluster penalty deliberately weak.
    dense_cluster_penalty = np.exp(-0.12 * overlap)

    # Future potential: each endpoint should retain useful alternatives after
    # selecting this edge. The diagonal is excluded from continuation evidence.
    continuation = local_quality.copy()
    np.fill_diagonal(continuation, 0.0)

    continuation_sums = continuation.sum(axis=1, keepdims=True)
    continuation_prob = np.divide(
        continuation,
        continuation_sums,
        out=np.zeros_like(continuation),
        where=continuation_sums > 0.0,
    )

    endpoint_future = np.sqrt(
        np.maximum(continuation_prob.sum(axis=1), eps)[:, None]
        * np.maximum(continuation_prob.sum(axis=0), eps)[None, :]
    )

    # Prefer edges whose endpoints have multiple plausible continuations.
    future_potential = 0.88 + 0.24 * np.sqrt(
        np.clip(endpoint_future, 0.0, 1.0)
    )

    # Penalize edges that consume an unusually scarce endpoint option only mildly.
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
    urgency = 1.0 + 0.08 * scarcity

    core = (
        local_quality
        * endpoint_compatibility
        * rank_quality
        * bridge_potential
        * dense_cluster_penalty
        * future_potential
        * urgency[:, None]
        * urgency[None, :]
    )
    core[~valid] = 0.0

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

    # Tight cap protects small and sparse instances from excessive exploration.
    global_mix = np.clip(0.10 + 0.16 * concentration, 0.10, 0.26)

    score = (
        (1.0 - global_mix[:, None]) * core
        + global_mix[:, None] * global_prob
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )
