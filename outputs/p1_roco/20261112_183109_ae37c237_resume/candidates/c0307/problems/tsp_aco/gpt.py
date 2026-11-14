import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized stochastic priorities for TSP edges."""
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
    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else float(np.median(d[valid]))
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(5, n - 1)
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)
    nearest_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)

    local_scale = np.divide(
        nearest_sum,
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
    global_quality = np.exp(-np.clip(global_ratio, 0.0, 50.0) / 1.50)

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
    endpoint_quality = np.sqrt(
        np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.15)
        * np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.15)
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
        (1.0 + 0.065 * np.maximum(row_rank - 1.0, 0.0))
        * (1.0 + 0.065 * np.maximum(col_rank - 1.0, 0.0))
    )

    threshold = np.maximum(1.35 * local_scale, 0.75 * global_scale)
    candidates = valid & (d <= threshold[:, None])

    candidate_float = candidates.astype(float)
    row_options = candidate_float.sum(axis=1)
    col_options = candidate_float.sum(axis=0)

    safe_row_options = np.maximum(row_options, 1.0)
    safe_col_options = np.maximum(col_options, 1.0)

    row_overlap = candidate_float @ candidate_float.T
    col_overlap = candidate_float.T @ candidate_float

    row_denominator = np.sqrt(
        safe_row_options[:, None] * safe_row_options[None, :]
    )
    col_denominator = np.sqrt(
        safe_col_options[:, None] * safe_col_options[None, :]
    )

    overlap = 0.5 * (
        np.divide(
            row_overlap,
            row_denominator,
            out=np.zeros_like(row_overlap),
            where=row_denominator > 0.0,
        )
        + np.divide(
            col_overlap,
            col_denominator,
            out=np.zeros_like(col_overlap),
            where=col_denominator > 0.0,
        )
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge = 1.0 + 0.10 / np.sqrt(
        safe_row_options[:, None] * safe_col_options[None, :]
    )
    cluster_penalty = 1.0 - 0.09 * overlap

    continuation = local_quality.copy()
    np.fill_diagonal(continuation, 0.0)
    continuation_sums = continuation.sum(axis=1, keepdims=True)

    remaining_row = continuation_sums - continuation
    remaining_col = continuation_sums.T - continuation.T

    row_future = np.divide(
        remaining_row,
        np.maximum(continuation_sums, eps),
        out=np.zeros_like(remaining_row),
        where=continuation_sums > 0.0,
    )
    col_future = np.divide(
        remaining_col,
        np.maximum(continuation_sums.T, eps),
        out=np.zeros_like(remaining_col),
        where=continuation_sums.T > 0.0,
    )

    future_potential = 0.90 + 0.20 * np.sqrt(
        np.clip(row_future * col_future, 0.0, 1.0)
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
    urgency = 1.0 + 0.08 * scarcity

    core = (
        local_quality
        * endpoint_quality
        * rank_quality
        * bridge
        * cluster_penalty
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
        np.where(
            local_prob > 0.0,
            local_prob * np.log(np.maximum(local_prob, eps)),
            0.0,
        ),
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

    global_mix = np.clip(0.10 + 0.15 * concentration, 0.10, 0.25)

    score = (
        (1.0 - global_mix[:, None]) * core
        + global_mix[:, None] * global_prob
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    positive = score > 0.0
    row_max = np.max(np.where(positive, score, 0.0), axis=1, keepdims=True)
    floor = 0.008 * row_max
    score = np.where(valid, np.maximum(score, floor), 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
