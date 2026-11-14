import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive row-normalized edge priorities for stochastic TSP sampling."""
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
    work = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    k = min(5, n - 1)
    sorted_rows = np.sort(work, axis=1)
    nearest = sorted_rows[:, :k]
    nearest_valid = np.isfinite(nearest)
    nearest_count = nearest_valid.sum(axis=1)

    local_scale = np.divide(
        np.where(nearest_valid, nearest, 0.0).sum(axis=1),
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    pair_scale = np.sqrt(np.maximum(row_scale * col_scale, eps))

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

    local_quality = np.exp(-np.clip(local_ratio, 0.0, 50.0) / 1.05)
    global_quality = np.exp(-np.clip(global_ratio, 0.0, 50.0) / 1.60)

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.10)
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.10)
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_sums = local_quality.sum(axis=1, keepdims=True)
    local_prob = np.divide(
        local_quality,
        row_sums,
        out=np.zeros_like(local_quality),
        where=row_sums > 0.0,
    )

    entropy = -np.sum(
        np.where(local_prob > 0.0, local_prob * np.log(local_prob), 0.0),
        axis=1,
    )
    valid_counts = valid.sum(axis=1).astype(float)
    entropy_reference = np.log(np.maximum(valid_counts, 2.0))
    entropy_level = np.divide(
        entropy,
        entropy_reference,
        out=np.zeros(n, dtype=float),
        where=entropy_reference > 0.0,
    )
    entropy_level = np.clip(entropy_level, 0.0, 1.0)

    edge_mix = 0.10 + 0.16 * (
        entropy_level[:, None] + entropy_level[None, :]
    ) / 2.0
    distance_quality = (
        (1.0 - edge_mix) * local_quality
        + edge_mix * global_quality
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.argsort(row_order, axis=1, kind="stable").astype(float) + 1.0

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.argsort(col_order, axis=0, kind="stable").astype(float) + 1.0

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.07 * np.maximum(row_rank - 1.0, 0.0))
        * (1.0 + 0.07 * np.maximum(col_rank - 1.0, 0.0))
    )

    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = np.argsort(d[i, candidates], kind="stable")
            neighborhood[i, candidates[order[:k]]] = 1.0

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

    bridge_bonus = 1.0 + 0.14 * (1.0 - overlap)
    cluster_penalty = np.exp(-0.07 * overlap)

    residual_row = np.maximum(row_sums - local_quality, 0.0)
    residual_row_prob = np.divide(
        residual_row,
        np.maximum(residual_row.sum(axis=1, keepdims=True), eps),
        out=np.zeros_like(residual_row),
        where=residual_row.sum(axis=1, keepdims=True) > 0.0,
    )
    residual_row_entropy = -np.sum(
        np.where(
            residual_row_prob > 0.0,
            residual_row_prob * np.log(residual_row_prob),
            0.0,
        ),
        axis=1,
    )
    row_effective = np.exp(np.clip(residual_row_entropy, 0.0, 20.0))
    row_diversity = np.clip(
        row_effective / np.maximum(valid_counts - 1.0, 1.0),
        0.0,
        1.0,
    )
    row_mass = np.divide(
        residual_row.sum(axis=1),
        np.maximum(row_sums[:, 0], eps),
    )
    row_reserve = 0.55 * row_diversity + 0.45 * row_mass

    incoming_quality = local_quality.T
    incoming_sums = incoming_quality.sum(axis=1, keepdims=True)
    residual_col = np.maximum(incoming_sums - incoming_quality, 0.0)
    residual_col_prob = np.divide(
        residual_col,
        np.maximum(residual_col.sum(axis=1, keepdims=True), eps),
        out=np.zeros_like(residual_col),
        where=residual_col.sum(axis=1, keepdims=True) > 0.0,
    )
    residual_col_entropy = -np.sum(
        np.where(
            residual_col_prob > 0.0,
            residual_col_prob * np.log(residual_col_prob),
            0.0,
        ),
        axis=1,
    )
    col_effective = np.exp(np.clip(residual_col_entropy, 0.0, 20.0))
    incoming_counts = valid.sum(axis=0).astype(float)
    col_diversity = np.clip(
        col_effective / np.maximum(incoming_counts - 1.0, 1.0),
        0.0,
        1.0,
    )
    col_mass = np.divide(
        residual_col.sum(axis=1),
        np.maximum(incoming_sums[:, 0], eps),
    )
    col_reserve = 0.55 * col_diversity + 0.45 * col_mass

    future_potential = 0.84 + 0.22 * np.sqrt(
        np.clip(row_reserve[:, None] * col_reserve[None, :], 0.0, 1.0)
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

    exploratory_floor = np.clip(
        0.035 + 0.065 * (1.0 - entropy_level),
        0.035,
        0.10,
    )[:, None]
    uniform_valid = np.divide(
        valid.astype(float),
        np.maximum(valid_counts[:, None], 1.0),
    )
    exploratory_prob = (
        (1.0 - exploratory_floor) * local_prob
        + exploratory_floor * uniform_valid
    )

    global_mix = np.clip(
        0.08
        + 0.12 * (1.0 - entropy_level)
        + 0.04 * (1.0 - row_diversity),
        0.08,
        0.22,
    )[:, None]

    core = (
        distance_quality
        * mutual_quality
        * rank_quality
        * bridge_bonus
        * cluster_penalty
        * future_potential
        * urgency[:, None]
    )
    core[~valid] = 0.0

    score = (
        (1.0 - global_mix) * core
        + global_mix * exploratory_prob
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.where(
        valid,
        np.maximum(score, 0.02 * row_max),
        0.0,
    )

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
