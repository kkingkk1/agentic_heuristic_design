import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term-potential TSP edge indicators."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    # Use the union of directed valid edges.
    valid = finite_nonnegative | finite_nonnegative.T
    if not np.any(valid):
        return result

    # Robust symmetric distance construction.
    sym_d = np.zeros_like(d, dtype=float)

    both = finite_nonnegative & finite_nonnegative.T
    forward_only = finite_nonnegative & ~finite_nonnegative.T
    backward_only = finite_nonnegative.T & ~finite_nonnegative

    sym_d[both] = 0.5 * (d[both] + d.T[both])
    sym_d[forward_only] = d[forward_only]
    sym_d[backward_only] = d.T[backward_only]

    eps = np.finfo(float).eps
    positive = sym_d[valid & (sym_d > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    work = np.where(valid, sym_d, np.inf)
    sorted_distances = np.sort(work, axis=1)

    k = min(max(3, int(np.sqrt(n))), n - 1)

    local_scale = np.full(n, global_scale, dtype=float)
    best = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)
    local_count = np.zeros(n, dtype=float)

    for i in range(n):
        finite_row = sorted_distances[i][np.isfinite(sorted_distances[i])]

        if finite_row.size:
            local_count[i] = float(finite_row.size)
            best[i] = finite_row[0]

            if finite_row.size > 1:
                second[i] = finite_row[1]
            else:
                second[i] = finite_row[0]

            neighborhood_values = finite_row[: min(k, finite_row.size)]
            local_scale[i] = max(
                float(np.median(neighborhood_values)),
                eps,
            )

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(np.isfinite(second), second, safe_best)

    # High regret means the endpoint has few similarly good alternatives.
    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )

    # Distance quality at both global and endpoint-local scales.
    global_quality = np.exp(
        -np.clip(sym_d / global_scale, 0.0, 60.0)
    )

    row_ratio = np.divide(
        sym_d,
        local_scale[:, None],
        out=np.full_like(sym_d, np.inf),
        where=valid,
    )
    col_ratio = np.divide(
        sym_d,
        local_scale[None, :],
        out=np.full_like(sym_d, np.inf),
        where=valid,
    )

    local_quality = np.exp(
        -0.5 * np.clip(row_ratio + col_ratio, 0.0, 60.0)
    )

    # Stable local rank quality.
    row_order = np.argsort(row_ratio, axis=1, kind="stable")
    row_rank = np.empty_like(row_order)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]

    col_order = np.argsort(col_ratio, axis=0, kind="stable")
    col_rank = np.empty_like(col_order)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    rank_quality = 1.0 / np.sqrt(
        1.0 + row_rank.astype(float) + col_rank.astype(float)
    )

    # Weighted option mass estimates how much future choice remains at
    # each endpoint. Sparse endpoints receive a restrained reserve bonus.
    affinity = np.zeros_like(sym_d, dtype=float)
    affinity[valid] = np.exp(
        -np.clip(sym_d[valid] / global_scale, 0.0, 60.0)
    )

    option_mass = affinity.sum(axis=1)
    option_mass = np.maximum(option_mass, 1.0)

    scarcity = 1.0 / np.sqrt(option_mass[:, None] * option_mass[None, :])
    scarcity_scale = np.median(scarcity[valid]) if np.any(valid) else 1.0
    scarcity_scale = max(float(scarcity_scale), eps)

    reserve_factor = 1.0 + 0.24 * np.tanh(
        scarcity / scarcity_scale
    )

    urgency = np.sqrt(
        (1.0 + 0.70 * regret[:, None])
        * (1.0 + 0.70 * regret[None, :])
    )

    # Build compact endpoint neighborhoods.
    neighborhood = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])

        if candidates.size:
            order = candidates[
                np.argsort(sym_d[i, candidates], kind="stable")
            ]
            selected = order[:k]
            neighborhood[i, selected] = 1.0

    degree = neighborhood.sum(axis=1)
    degree = np.maximum(degree, 1.0)

    shared = neighborhood @ neighborhood.T

    overlap = shared / np.minimum(
        degree[:, None],
        degree[None, :],
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    # Bridge reward favors cross-region edges without overwhelming distance.
    bridge_factor = 1.0 + 0.28 * (1.0 - overlap)

    # Two-hop support estimates whether an endpoint can reach other local
    # neighborhoods through the candidate graph.
    two_hop = neighborhood @ neighborhood.T
    support = two_hop.sum(axis=1)

    support_scale = np.maximum(
        np.percentile(support[support > 0.0], 75)
        if np.any(support > 0.0)
        else 1.0,
        1.0,
    )

    low_support = np.clip(
        1.0 - support / support_scale,
        0.0,
        1.0,
    )

    access_factor = 1.0 + 0.16 * np.sqrt(
        low_support[:, None] * low_support[None, :]
    )

    # Mutual nearest edges remain useful, but receive only a limited bonus.
    nearest = valid & (
        sym_d <= safe_best[:, None] + 1e-12 * max(global_scale, 1.0)
    )
    mutual_nearest = nearest & nearest.T
    nearest_factor = 1.0 + 0.12 * mutual_nearest.astype(float)

    raw_score = (
        global_quality
        * local_quality
        * rank_quality
        * urgency
        * reserve_factor
        * bridge_factor
        * access_factor
        * nearest_factor
    )

    raw_score[~valid] = 0.0
    np.fill_diagonal(raw_score, 0.0)
    raw_score = 0.5 * (raw_score + raw_score.T)

    if not np.all(np.isfinite(raw_score)):
        raw_score = np.where(np.isfinite(raw_score), raw_score, 0.0)

    total = float(raw_score.sum())

    if total <= 0.0:
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / count
        return result

    normalized = raw_score / total

    # One-pass node-load balancing prevents a few dense nodes from absorbing
    # most probability mass while retaining the learned edge contrast.
    node_load = normalized.sum(axis=1)
    positive_load = node_load > 0.0

    if np.any(positive_load):
        target_load = 1.0 / float(np.count_nonzero(positive_load))
        balance = np.ones(n, dtype=float)

        balance[positive_load] = np.sqrt(
            target_load / np.maximum(node_load[positive_load], eps)
        )

        balance = np.clip(balance, 0.55, 1.80)
        balanced = normalized * balance[:, None] * balance[None, :]
        balanced[~valid] = 0.0
        balanced = 0.5 * (balanced + balanced.T)

        balanced_total = float(balanced.sum())
        if np.isfinite(balanced_total) and balanced_total > 0.0:
            normalized = balanced / balanced_total

    # Adaptive exploration floor based on concentration.
    positive_values = normalized[valid]
    valid_count = float(positive_values.size)

    if valid_count <= 1.0:
        result[valid] = 1.0
        return result

    concentration = float(np.sum(positive_values * positive_values))
    concentration_ratio = np.clip(
        concentration * valid_count,
        1.0,
        valid_count,
    )

    concentration_level = (
        concentration_ratio - 1.0
    ) / max(valid_count - 1.0, 1.0)

    floor_strength = 0.012 + 0.038 * concentration_level
    floor = floor_strength / valid_count

    result = (1.0 - floor_strength) * normalized
    result[valid] += floor
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    final_total = float(result.sum())

    if not np.isfinite(final_total) or final_total <= 0.0:
        result[:] = 0.0
        result[valid] = 1.0 / valid_count
    else:
        result /= final_total

    return result
