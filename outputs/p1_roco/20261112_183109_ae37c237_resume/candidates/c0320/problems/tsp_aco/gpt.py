import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    valid_counts = valid.sum(axis=1)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    positive_values = d[valid & (d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    k = min(4, n - 1)
    sorted_rows = np.sort(work, axis=1)
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)

    local_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    local_scale = np.divide(
        local_sum,
        np.maximum(finite_nearest.sum(axis=1), 1),
        out=np.full(n, global_scale, dtype=float),
        where=finite_nearest.sum(axis=1) > 0,
    )
    local_scale = np.maximum(
        0.5 * local_scale + 0.5 * global_scale,
        eps,
    )

    robust_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = np.divide(
        d,
        robust_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    base_quality = np.exp(
        -np.clip(normalized_distance, 0.0, 40.0) / 0.95
    )
    row_quality = np.exp(-np.clip(row_ratio, 0.0, 40.0) / 1.05)
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 40.0) / 1.05)
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(work, axis=1)
    row_rank_index = np.empty_like(row_order)
    row_rank_index[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]
    row_ranks = row_rank_index.astype(float) + 1.0

    col_order = np.argsort(work, axis=0)
    col_rank_index = np.empty_like(col_order)
    col_rank_index[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]
    col_ranks = col_rank_index.astype(float) + 1.0

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.12 * row_ranks) * (1.0 + 0.12 * col_ranks)
    )
    rank_quality[~valid] = 0.0

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(np.abs(safe_second), global_scale * eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.16 * regret

    neighborhood = np.zeros_like(d, dtype=float)
    nearest_indices = np.argsort(work, axis=1)[:, :k]
    for i in range(n):
        usable = valid[i, nearest_indices[i]]
        neighborhood[i, nearest_indices[i][usable]] = 1.0

    shared_neighborhood = neighborhood @ neighborhood.T
    neighborhood_size = np.maximum(neighborhood.sum(axis=1), 1.0)
    overlap = shared_neighborhood / np.sqrt(
        neighborhood_size[:, None] * neighborhood_size[None, :]
    )

    support = mutual_quality * rank_quality
    residual_diversity = support * (1.0 - np.clip(overlap, 0.0, 1.0))

    row_competition = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )
    scarcity = 1.0 + 0.07 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    correction = (
        (1.0 + 0.045 * np.clip(residual_diversity, 0.0, 1.0))
        * scarcity
        * urgency[:, None]
        * urgency[None, :]
    )

    score = base_quality * mutual_quality * rank_quality * correction
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    base_probability = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    entropy = -np.sum(
        np.where(
            base_probability > 0.0,
            base_probability * np.log(np.maximum(base_probability, eps)),
            0.0,
        ),
        axis=1,
    )
    max_entropy = np.log(np.maximum(valid_counts, 2))
    normalized_entropy = np.divide(
        entropy,
        max_entropy,
        out=np.zeros_like(entropy),
        where=max_entropy > 0.0,
    )

    valid_uniform = np.divide(
        valid.astype(float),
        np.maximum(valid_counts[:, None], 1),
    )

    floor = 0.015 + 0.035 * (1.0 - np.clip(normalized_entropy, 0.0, 1.0))
    mix = 0.08 + 0.22 * (1.0 - np.clip(normalized_entropy, 0.0, 1.0))

    result = (
        (1.0 - mix[:, None]) * base_probability
        + mix[:, None] * (
            (1.0 - floor[:, None]) * valid_uniform
            + floor[:, None] * valid_uniform
        )
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    final_totals = result.sum(axis=1, keepdims=True)
    return np.divide(
        result,
        final_totals,
        out=np.zeros_like(result),
        where=final_totals > 0.0,
    )
