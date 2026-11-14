import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diversity-aware long-term edge priorities for stochastic TSP sampling."""
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
    values = d[valid]
    positive = values[values > 0.0]

    global_scale = (
        float(np.median(positive))
        if positive.size
        else float(np.median(values))
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

    local_scale = np.divide(
        np.where(finite_nearest, nearest, 0.0).sum(axis=1),
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

    endpoint_compatibility = np.sqrt(
        np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.20)
        * np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.20)
    )

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

    # Reward edges that connect locally different regions.
    bridge_bonus = 1.0 + 0.18 * (1.0 - overlap)

    # Keep dense-cluster suppression deliberately mild.
    cluster_penalty = np.exp(-0.10 * overlap)

    row_sums = local_quality.sum(axis=1, keepdims=True)
    row_prob = np.divide(
        local_quality,
        row_sums,
        out=np.zeros_like(local_quality),
        where=row_sums > 0.0,
    )

    # Adaptive probability floor: preserve global alternatives without
    # overwhelming strong local edges.
    valid_counts = valid.sum(axis=1, keepdims=True).astype(float)
    uniform_valid = np.divide(
        valid.astype(float),
        np.maximum(valid_counts, 1.0),
    )

    entropy = -np.sum(
        np.where(row_prob > 0.0, row_prob * np.log(row_prob), 0.0),
        axis=1,
    )
    entropy_ref = np.log(np.maximum(valid_counts[:, 0], 2.0))
    concentration = np.clip(
        1.0 - entropy / np.maximum(entropy_ref, eps),
        0.0,
        1.0,
    )

    floor_strength = np.clip(
        0.035 + 0.075 * concentration,
        0.035,
        0.11,
    )[:, None]

    exploratory_prob = (
        (1.0 - floor_strength) * row_prob
        + floor_strength * uniform_valid
    )

    # Long-term potential:
    # remove the current edge and measure remaining mass, entropy, and
    # effective candidate count at both directed tour endpoints.
    residual_mass = np.maximum(row_sums - local_quality, 0.0)
    residual_prob = np.divide(
        np.maximum(local_quality - np.diag(local_quality)[:, None], 0.0),
        np.maximum(residual_mass, eps),
        out=np.zeros_like(local_quality),
        where=residual_mass > 0.0,
    )

    residual_entropy = -np.sum(
        np.where(
            residual_prob > 0.0,
            residual_prob * np.log(residual_prob),
            0.0,
        ),
        axis=1,
    )
    effective_count = np.exp(np.clip(residual_entropy, 0.0, 20.0))

    candidate_count = np.maximum(valid_counts[:, 0] - 1.0, 1.0)
    diversity_ratio = np.clip(effective_count / candidate_count, 0.0, 1.0)
    mass_ratio = np.divide(
        residual_mass[:, 0],
        np.maximum(row_sums[:, 0], eps),
    )
    endpoint_reserve = 0.55 * diversity_ratio + 0.45 * mass_ratio

    # For i -> j, i needs remaining outgoing choices and j needs remaining
    # incoming choices. The incoming reserve is approximated by column evidence.
    incoming_quality = local_quality.T
    incoming_sums = incoming_quality.sum(axis=1, keepdims=True)
    incoming_residual = np.maximum(
        incoming_sums - incoming_quality,
        0.0,
    )
    incoming_prob = np.divide(
        incoming_residual,
        np.maximum(incoming_residual.sum(axis=1, keepdims=True), eps),
        out=np.zeros_like(incoming_residual),
        where=incoming_residual.sum(axis=1, keepdims=True) > 0.0,
    )
    incoming_entropy = -np.sum(
        np.where(
            incoming_prob > 0.0,
            incoming_prob * np.log(incoming_prob),
            0.0,
        ),
        axis=1,
    )
    incoming_effective = np.exp(np.clip(incoming_entropy, 0.0, 20.0))
    incoming_count = np.maximum(valid.sum(axis=0).astype(float) - 1.0, 1.0)
    incoming_diversity = np.clip(
        incoming_effective / incoming_count,
        0.0,
        1.0,
    )
    incoming_mass = np.divide(
        incoming_residual.sum(axis=1),
        np.maximum(incoming_sums[:, 0], eps),
    )
    incoming_reserve = 0.55 * incoming_diversity + 0.45 * incoming_mass

    future_potential = (
        0.78
        + 0.28
        * np.sqrt(
            np.clip(
                endpoint_reserve[:, None] * incoming_reserve[None, :],
                0.0,
                1.0,
            )
        )
    )

    # Mildly prioritize edges that are unusually important to scarce endpoints.
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
    urgency = 1.0 + 0.10 * scarcity

    rank_rows = np.argsort(np.argsort(work, axis=1), axis=1)
    rank_cols = np.argsort(np.argsort(work, axis=0), axis=0)
    row_rank = np.where(valid, rank_rows + 1.0, 0.0)
    col_rank = np.where(valid, rank_cols + 1.0, 0.0)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.06 * row_rank)
        * (1.0 + 0.06 * col_rank)
    )

    core = (
        local_quality
        * endpoint_compatibility
        * rank_quality
        * bridge_bonus
        * cluster_penalty
        * future_potential
        * urgency[:, None]
    )
    core[~valid] = 0.0

    # Strongly concentrated rows receive more exploration; already diverse
    # rows retain a larger share of their local evidence.
    global_mix = np.clip(
        0.08
        + 0.13 * concentration
        + 0.05 * (1.0 - diversity_ratio),
        0.08,
        0.24,
    )[:, None]

    score = (
        (1.0 - global_mix) * core
        + global_mix * exploratory_prob
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
