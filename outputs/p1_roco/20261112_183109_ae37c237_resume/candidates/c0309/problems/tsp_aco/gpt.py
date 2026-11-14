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
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(4, n - 1)
    nearest = sorted_rows[:, :k]
    nearest_valid = np.isfinite(nearest)

    nearest_sum = np.where(nearest_valid, nearest, 0.0).sum(axis=1)
    nearest_count = nearest_valid.sum(axis=1)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
    )
    local_scale = np.where(local_scale > eps, local_scale, global_scale)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(np.maximum(row_scale * col_scale, eps))

    safe_d = np.where(valid, d, np.inf)

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

    # Estimate local uncertainty from normalized inverse-distance entropy.
    inverse_local = np.where(
        valid,
        np.exp(-np.clip(row_ratio, 0.0, 50.0)),
        0.0,
    )
    inverse_sums = inverse_local.sum(axis=1, keepdims=True)

    local_prob = np.divide(
        inverse_local,
        inverse_sums,
        out=np.zeros_like(inverse_local),
        where=inverse_sums > 0.0,
    )

    entropy = -np.sum(
        np.where(local_prob > 0.0, local_prob * np.log(local_prob), 0.0),
        axis=1,
    )
    valid_count = valid.sum(axis=1)
    entropy_denominator = np.log(np.maximum(valid_count, 2))
    entropy_level = np.divide(
        entropy,
        entropy_denominator,
        out=np.zeros_like(entropy),
        where=entropy_denominator > 0.0,
    )
    entropy_level = np.clip(entropy_level, 0.0, 1.0)

    # Use more global information only when the local neighborhood is uncertain.
    row_global_mix = 0.10 + 0.18 * entropy_level
    edge_global_mix = 0.5 * (
        row_global_mix[:, None] + row_global_mix[None, :]
    )
    distance_quality = (
        (1.0 - edge_global_mix) * local_quality
        + edge_global_mix * global_quality
    )

    # Bidirectional rank quality.
    row_order = np.argsort(safe_d, axis=1)
    row_ranks = np.empty_like(row_order, dtype=float)
    row_ranks[np.arange(n)[:, None], row_order] = (
        np.arange(n, dtype=float)[None, :] + 1.0
    )

    col_order = np.argsort(safe_d, axis=0)
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

    # Ensure every valid row has a usable candidate set.
    candidate_counts = candidates.sum(axis=1)
    missing_rows = candidate_counts == 0
    if np.any(missing_rows):
        candidates[missing_rows] = valid[missing_rows]
        candidate_counts = candidates.sum(axis=1)

    candidate_float = candidates.astype(float)
    row_options = candidate_counts.astype(float)
    col_options = candidates.sum(axis=0).astype(float)

    safe_row_options = np.maximum(row_options, 1.0)
    safe_col_options = np.maximum(col_options, 1.0)

    # Mild bridge reward for endpoints with fewer alternatives.
    bridge = 1.0 + 0.08 / np.sqrt(
        safe_row_options[:, None] * safe_col_options[None, :]
    )

    # Future extensibility after excluding the current edge.
    remaining_row = np.maximum(
        row_options[:, None] - candidate_float,
        0.0,
    )
    remaining_col = np.maximum(
        col_options[None, :] - candidate_float,
        0.0,
    )

    max_remaining = max(float(n - 2), 1.0)
    remaining_support = 0.5 * (
        np.sqrt(remaining_row / max_remaining)
        + np.sqrt(remaining_col / max_remaining)
    )

    # Reward forced or nearly forced endpoints, but keep the effect conservative.
    future_factor = 0.96 + 0.10 * (1.0 - remaining_support)

    # Remaining candidate quality, excluding the current edge.
    residual_row_quality = np.maximum(
        candidate_float * local_quality - candidate_float * local_quality,
        0.0,
    )
    del residual_row_quality

    candidate_quality = candidate_float * local_quality
    row_quality_sum = candidate_quality.sum(axis=1, keepdims=True)
    residual_quality_i = np.maximum(
        row_quality_sum - candidate_quality,
        0.0,
    )
    col_quality_sum = candidate_quality.sum(axis=0, keepdims=True)
    residual_quality_j = np.maximum(
        col_quality_sum - candidate_quality,
        0.0,
    )

    normalized_residual_i = np.divide(
        residual_quality_i,
        np.maximum(row_quality_sum, eps),
        out=np.zeros_like(residual_quality_i),
        where=row_quality_sum > 0.0,
    )
    normalized_residual_j = np.divide(
        residual_quality_j,
        np.maximum(col_quality_sum, eps),
        out=np.zeros_like(residual_quality_j),
        where=col_quality_sum > 0.0,
    )

    residual_factor = 0.97 + 0.06 * np.sqrt(
        normalized_residual_i * normalized_residual_j
    )

    score = (
        distance_quality
        * mutual_quality
        * rank_quality
        * bridge
        * future_factor
        * residual_factor
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    positive = score > 0.0
    row_max = np.max(np.where(positive, score, 0.0), axis=1, keepdims=True)

    # Slightly stronger floor to preserve exploratory alternatives.
    floor = 0.025 * row_max
    score = np.where(valid, np.maximum(score, floor), 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
