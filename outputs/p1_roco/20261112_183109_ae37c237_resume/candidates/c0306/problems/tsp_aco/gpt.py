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

    positive_values = d[valid & (d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(4, n - 1)
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)
    nearest_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    nearest_count = finite_nearest.sum(axis=1)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
    )
    local_scale = np.where(
        local_scale > eps,
        local_scale,
        global_scale,
    )

    second = sorted_rows[:, 1] if n > 2 else sorted_rows[:, 0]
    best = sorted_rows[:, 0]

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(np.abs(safe_second), eps),
        0.0,
        1.0,
    )

    urgency = 1.0 + 0.10 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    local_ratio = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    local_quality = np.exp(-np.clip(local_ratio, 0.0, 50.0) / 1.05)

    global_ratio = np.divide(
        d,
        global_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    global_quality = np.exp(-np.clip(global_ratio, 0.0, 50.0) / 1.60)

    # Keep the global contribution bounded so local short edges remain dominant.
    global_mix = 0.18
    distance_quality = (
        (1.0 - global_mix) * local_quality
        + global_mix * global_quality
    )

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

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.10)
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.10)
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(np.where(valid, d, np.inf), axis=1)
    row_ranks = np.empty_like(row_order, dtype=float)
    row_ranks[np.arange(n)[:, None], row_order] = (
        np.arange(n, dtype=float)[None, :] + 1.0
    )

    col_order = np.argsort(np.where(valid, d, np.inf), axis=0)
    col_ranks = np.empty_like(col_order, dtype=float)
    col_ranks[col_order, np.arange(n)[None, :]] = (
        np.arange(n, dtype=float)[:, None] + 1.0
    )

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.075 * np.maximum(row_ranks - 1.0, 0.0))
        * (1.0 + 0.075 * np.maximum(col_ranks - 1.0, 0.0))
    )

    candidate_threshold = np.maximum(
        1.35 * local_scale,
        0.75 * global_scale,
    )
    candidates = valid & (d <= candidate_threshold[:, None])

    row_options = candidates.sum(axis=1).astype(float)
    col_options = candidates.sum(axis=0).astype(float)

    safe_row_options = np.maximum(row_options, 1.0)
    safe_col_options = np.maximum(col_options, 1.0)

    # Mild bridge reward for endpoints with relatively few local alternatives.
    bridge = 1.0 + 0.08 / np.sqrt(
        safe_row_options[:, None] * safe_col_options[None, :]
    )

    # Symmetric overlap estimate from outgoing and incoming neighborhoods.
    candidate_float = candidates.astype(float)
    row_overlap = candidate_float @ candidate_float.T
    col_overlap = candidate_float.T @ candidate_float

    row_denominator = np.sqrt(
        safe_row_options[:, None] * safe_row_options[None, :]
    )
    col_denominator = np.sqrt(
        safe_col_options[:, None] * safe_col_options[None, :]
    )

    overlap = 0.5 * (
        np.divide(row_overlap, row_denominator, out=np.zeros_like(row_overlap))
        + np.divide(col_overlap, col_denominator, out=np.zeros_like(col_overlap))
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    # Cap the penalty to preserve useful short edges inside dense regions.
    cluster_penalty = 1.0 - 0.08 * overlap

    score = (
        distance_quality
        * mutual_quality
        * rank_quality
        * bridge
        * cluster_penalty
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    positive = score > 0.0
    row_max = np.max(np.where(positive, score, 0.0), axis=1, keepdims=True)
    floor = 0.015 * row_max
    score = np.where(valid, np.maximum(score, floor), 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
