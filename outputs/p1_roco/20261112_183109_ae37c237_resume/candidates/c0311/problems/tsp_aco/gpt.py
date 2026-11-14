import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term-aware edge priorities for stochastic TSP sampling."""
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

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)
    valid_counts_row = valid.sum(axis=1).astype(float)
    valid_counts_col = valid.sum(axis=0).astype(float)

    max_k = max(1, n - 1)
    k = min(5, max_k)

    sorted_rows = np.sort(work, axis=1)
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)
    nearest_count = finite_nearest.sum(axis=1)

    local_scale = np.divide(
        np.where(finite_nearest, nearest, 0.0).sum(axis=1),
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
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

    local_quality = np.exp(-np.clip(local_ratio, 0.0, 60.0) / 1.15)
    global_quality = np.exp(-np.clip(global_ratio, 0.0, 60.0) / 1.80)

    # Blend local and global evidence continuously.
    quality = np.sqrt(local_quality * global_quality)
    quality[~valid] = 0.0

    # Soft neighborhood structure for bridge diversity.
    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = np.argsort(d[i, candidates], kind="stable")
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

    # Bridges receive a bounded bonus; dense-cluster suppression remains mild.
    bridge_bonus = 1.0 + 0.16 * (1.0 - overlap)
    cluster_penalty = np.exp(-0.07 * overlap)

    row_mass = quality.sum(axis=1, keepdims=True)
    col_mass = quality.sum(axis=0, keepdims=True)

    row_prob = np.divide(
        quality,
        np.maximum(row_mass, eps),
        out=np.zeros_like(quality),
        where=row_mass > 0.0,
    )
    col_prob = np.divide(
        quality,
        np.maximum(col_mass, eps),
        out=np.zeros_like(quality),
        where=col_mass > 0.0,
    )

    row_entropy = -np.sum(
        np.where(row_prob > 0.0, row_prob * np.log(row_prob), 0.0),
        axis=1,
    )
    col_entropy = -np.sum(
        np.where(col_prob > 0.0, col_prob * np.log(col_prob), 0.0),
        axis=0,
    )

    row_entropy_ref = np.log(np.maximum(valid_counts_row, 2.0))
    col_entropy_ref = np.log(np.maximum(valid_counts_col, 2.0))

    row_concentration = np.clip(
        1.0 - row_entropy / np.maximum(row_entropy_ref, eps),
        0.0,
        1.0,
    )
    col_concentration = np.clip(
        1.0 - col_entropy / np.maximum(col_entropy_ref, eps),
        0.0,
        1.0,
    )

    # Explicitly remove each current edge and evaluate the remaining choices.
    row_residual_mass = np.maximum(row_mass - quality, 0.0)
    col_residual_mass = np.maximum(col_mass - quality, 0.0)

    quality_log = np.where(quality > 0.0, quality * np.log(quality), 0.0)
    row_log_mass = quality_log.sum(axis=1, keepdims=True)
    col_log_mass = quality_log.sum(axis=0, keepdims=True)

    row_residual_entropy = np.zeros_like(quality)
    col_residual_entropy = np.zeros_like(quality)

    row_positive = row_residual_mass > eps
    col_positive = col_residual_mass > eps

    row_residual_entropy[row_positive] = (
        np.log(np.maximum(row_residual_mass[row_positive], eps))
        - (
            row_log_mass - quality_log
        )[row_positive]
        / row_residual_mass[row_positive]
    )
    col_residual_entropy[col_positive] = (
        np.log(np.maximum(col_residual_mass[col_positive], eps))
        - (
            col_log_mass - quality_log
        )[col_positive]
        / col_residual_mass[col_positive]
    )

    row_effective = np.exp(np.clip(row_residual_entropy, 0.0, 20.0))
    col_effective = np.exp(np.clip(col_residual_entropy, 0.0, 20.0))

    row_remaining_count = np.maximum(valid_counts_row[:, None] - 1.0, 1.0)
    col_remaining_count = np.maximum(valid_counts_col[None, :] - 1.0, 1.0)

    row_diversity = np.clip(
        row_effective / row_remaining_count,
        0.0,
        1.0,
    )
    col_diversity = np.clip(
        col_effective / col_remaining_count,
        0.0,
        1.0,
    )

    row_mass_reserve = np.divide(
        row_residual_mass,
        np.maximum(row_mass, eps),
        out=np.zeros_like(quality),
        where=row_mass > 0.0,
    )
    col_mass_reserve = np.divide(
        col_residual_mass,
        np.maximum(col_mass, eps),
        out=np.zeros_like(quality),
        where=col_mass > 0.0,
    )

    row_reserve = 0.58 * row_diversity + 0.42 * row_mass_reserve
    col_reserve = 0.58 * col_diversity + 0.42 * col_mass_reserve

    future_potential = (
        0.76
        + 0.34
        * np.sqrt(
            np.clip(row_reserve * col_reserve, 0.0, 1.0)
        )
    )

    # Smooth rank attenuation avoids a hard top-k cutoff.
    row_rank = np.zeros_like(d, dtype=float)
    col_rank = np.zeros_like(d, dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[
                np.argsort(d[i, candidates], kind="stable")
            ]
            row_rank[i, order] = np.arange(1.0, candidates.size + 1.0)

    for j in range(n):
        candidates = np.flatnonzero(valid[:, j])
        if candidates.size:
            order = candidates[
                np.argsort(d[candidates, j], kind="stable")
            ]
            col_rank[order, j] = np.arange(1.0, candidates.size + 1.0)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.045 * row_rank)
        * (1.0 + 0.045 * col_rank)
    )

    # Scarcity bonus is endpoint-specific and deliberately bounded.
    row_best = np.min(work, axis=1)
    row_second = np.partition(work, 1, axis=1)[:, 1] if n > 2 else row_best
    row_best = np.where(np.isfinite(row_best), row_best, local_scale)
    row_second = np.where(
        np.isfinite(row_second),
        row_second,
        2.0 * row_best,
    )
    row_scarcity = np.clip(
        (row_second - row_best)
        / np.maximum(np.abs(row_second), eps),
        0.0,
        1.0,
    )

    col_best = np.min(work, axis=0)
    col_second = np.partition(work, 1, axis=0)[1, :] if n > 2 else col_best
    col_best = np.where(np.isfinite(col_best), col_best, local_scale)
    col_second = np.where(
        np.isfinite(col_second),
        col_second,
        2.0 * col_best,
    )
    col_scarcity = np.clip(
        (col_second - col_best)
        / np.maximum(np.abs(col_second), eps),
        0.0,
        1.0,
    )

    urgency = (
        1.0
        + 0.08 * row_scarcity[:, None]
        + 0.08 * col_scarcity[None, :]
    )

    core = (
        quality
        * rank_quality
        * bridge_bonus
        * cluster_penalty
        * future_potential
        * urgency
    )
    core[~valid] = 0.0

    # Adaptive floor preserves alternatives in concentrated rows and columns.
    row_floor = np.clip(
        0.035 + 0.085 * row_concentration,
        0.035,
        0.12,
    )[:, None]
    col_floor = np.clip(
        0.035 + 0.085 * col_concentration,
        0.035,
        0.12,
    )[None, :]

    uniform_row = np.divide(
        valid.astype(float),
        np.maximum(valid_counts_row[:, None], 1.0),
    )
    uniform_col = np.divide(
        valid.astype(float),
        np.maximum(valid_counts_col[None, :], 1.0),
    )

    exploratory = (
        0.5 * ((1.0 - row_floor) * row_prob + row_floor * uniform_row)
        + 0.5 * ((1.0 - col_floor) * col_prob + col_floor * uniform_col)
    )

    global_mix = np.clip(
        0.07
        + 0.12 * np.maximum(row_concentration[:, None], col_concentration[None, :])
        + 0.05
        * (
            1.0
            - np.sqrt(
                np.clip(row_diversity * col_diversity, 0.0, 1.0)
            )
        ),
        0.07,
        0.23,
    )

    score = (1.0 - global_mix) * core + global_mix * exploratory
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
