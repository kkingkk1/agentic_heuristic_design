import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, diverse, long-term-potential TSP edge indicators."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    valid = finite_nonnegative | finite_nonnegative.T
    if not np.any(valid):
        return result

    # Build a symmetric distance matrix from the union of directed edges.
    sym_d = np.zeros_like(d, dtype=float)

    both = finite_nonnegative & finite_nonnegative.T
    forward_only = finite_nonnegative & ~finite_nonnegative.T
    backward_only = finite_nonnegative.T & ~finite_nonnegative

    sym_d[both] = 0.5 * (d[both] + d.T[both])
    sym_d[forward_only] = d[forward_only]
    sym_d[backward_only] = d.T[backward_only]

    positive_distances = sym_d[valid & (sym_d > 0.0)]
    eps = np.finfo(float).eps

    if positive_distances.size:
        global_scale = float(np.median(positive_distances))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    work = np.where(valid, sym_d, np.inf)
    sorted_distances = np.sort(work, axis=1)

    valid_degree = valid.sum(axis=1).astype(float)
    max_degree = max(1.0, float(np.max(valid_degree)))

    # Keep neighborhoods compact, but reduce structural bonuses on tiny graphs.
    k = min(max(2, int(np.sqrt(n))), n - 1)
    graph_scale = min(1.0, max(0.35, (n - 1.0) / 6.0))

    local_scale = np.full(n, global_scale, dtype=float)
    best = np.full(n, global_scale, dtype=float)
    second = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = sorted_distances[i][np.isfinite(sorted_distances[i])]

        if row.size:
            best[i] = float(row[0])
            second[i] = float(row[1]) if row.size > 1 else float(row[0])

            local_values = row[: min(k, row.size)]
            local_scale[i] = max(float(np.median(local_values)), eps)

    # Choice regret: high values indicate that the best option is difficult
    # to replace, so the corresponding endpoint should be preserved carefully.
    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    global_quality = np.exp(
        -np.clip(np.divide(sym_d, global_scale), 0.0, 60.0)
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

    # Stable local ranks. Invalid edges are assigned the worst effective rank.
    row_order = np.argsort(row_ratio, axis=1, kind="stable")
    row_rank = np.empty_like(row_order)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]

    col_order = np.argsort(col_ratio, axis=0, kind="stable")
    col_rank = np.empty_like(col_order)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    rank_quality = 1.0 / np.sqrt(
        1.0 + row_rank.astype(float) + col_rank.astype(float)
    )

    # Affinity and endpoint option entropy estimate future flexibility.
    affinity = np.zeros_like(sym_d, dtype=float)
    affinity[valid] = np.exp(
        -np.clip(sym_d[valid] / global_scale, 0.0, 60.0)
    )

    affinity_mass = affinity.sum(axis=1)
    safe_mass = np.maximum(affinity_mass, eps)

    probabilities = np.divide(
        affinity,
        safe_mass[:, None],
        out=np.zeros_like(affinity),
        where=safe_mass[:, None] > 0.0,
    )

    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0),
        axis=1,
    )
    entropy_scale = np.log(np.maximum(valid_degree, 2.0))
    flexibility = np.divide(
        entropy,
        np.maximum(entropy_scale, 1.0),
        out=np.zeros(n, dtype=float),
        where=entropy_scale > 0.0,
    )
    flexibility = np.clip(flexibility, 0.0, 1.0)

    # Sparse endpoints receive a restrained reserve bonus. The cap depends on
    # graph size and therefore cannot dominate on tiny graphs.
    scarcity = 1.0 / np.sqrt(np.maximum(affinity_mass, eps))
    scarcity_reference = float(np.median(scarcity[valid_degree > 0]))
    scarcity_reference = max(scarcity_reference, eps)

    scarcity_bonus_cap = (0.10 + 0.12 * graph_scale)
    reserve_factor = 1.0 + scarcity_bonus_cap * np.tanh(
        scarcity[:, None] / scarcity_reference
    )

    urgency = np.sqrt(
        (1.0 + (0.55 + 0.20 * graph_scale) * regret[:, None])
        * (1.0 + (0.55 + 0.20 * graph_scale) * regret[None, :])
    )

    # Compact k-nearest-neighbor graph.
    neighborhood = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[
                np.argsort(sym_d[i, candidates], kind="stable")
            ]
            neighborhood[i, order[:k]] = 1.0

    # Mutual support is useful, but directed availability is retained in the
    # union graph. Symmetrization happens explicitly after construction.
    neighborhood = np.maximum(neighborhood, neighborhood.T)
    np.fill_diagonal(neighborhood, 0.0)

    neighborhood_degree = neighborhood.sum(axis=1)
    safe_neighborhood_degree = np.maximum(neighborhood_degree, 1.0)

    # Explicit two-hop calculation: remove direct/self paths so this is not
    # merely a duplicate of one-hop neighborhood overlap.
    two_hop_paths = neighborhood @ neighborhood
    direct_or_self = neighborhood + np.eye(n, dtype=float)
    two_hop_only = np.maximum(two_hop_paths - direct_or_self, 0.0)
    np.fill_diagonal(two_hop_only, 0.0)

    two_hop_reach = two_hop_only.sum(axis=1)
    reach_reference = float(
        np.percentile(two_hop_reach[two_hop_reach > 0.0], 75)
        if np.any(two_hop_reach > 0.0)
        else 1.0
    )
    reach_reference = max(reach_reference, 1.0)

    endpoint_access = np.clip(two_hop_reach / reach_reference, 0.0, 1.0)

    # Common-neighbor overlap identifies edges inside one region. Low overlap
    # indicates a possible bridge, but the reward is deliberately capped.
    shared_neighbors = neighborhood @ neighborhood.T
    overlap = shared_neighbors / np.minimum(
        safe_neighborhood_degree[:, None],
        safe_neighborhood_degree[None, :],
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_cap = (0.08 + 0.16 * graph_scale)
    bridge_factor = 1.0 + bridge_cap * (1.0 - overlap)

    # Reward edges whose endpoints both retain multiple future routes. This
    # discourages consuming a flexible hub too early.
    future_access = np.sqrt(
        (0.35 + 0.65 * endpoint_access[:, None])
        * (0.35 + 0.65 * endpoint_access[None, :])
    )

    # Preserve a small amount of direct-nearest-edge preference.
    row_best = np.min(np.where(valid, sym_d, np.inf), axis=1)
    nearest = valid & (
        sym_d <= row_best[:, None] + 1e-12 * max(global_scale, 1.0)
    )
    mutual_nearest = nearest & nearest.T
    nearest_factor = 1.0 + (0.06 + 0.05 * graph_scale) * (
        mutual_nearest.astype(float)
    )

    raw_score = (
        global_quality
        * local_quality
        * rank_quality
        * urgency
        * reserve_factor
        * bridge_factor
        * future_access
        * nearest_factor
    )

    raw_score[~valid] = 0.0
    np.fill_diagonal(raw_score, 0.0)
    raw_score = 0.5 * (raw_score + raw_score.T)

    raw_score = np.where(np.isfinite(raw_score), raw_score, 0.0)
    total = float(raw_score.sum())

    if total <= 0.0:
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / count
        return result

    normalized = raw_score / total

    # One-pass bounded node-load balancing.
    node_load = normalized.sum(axis=1)
    active = node_load > 0.0

    if np.any(active):
        target_load = 1.0 / float(np.count_nonzero(active))
        balance = np.ones(n, dtype=float)
        balance[active] = np.sqrt(
            target_load / np.maximum(node_load[active], eps)
        )
        balance = np.clip(balance, 0.65, 1.55)

        balanced = normalized * balance[:, None] * balance[None, :]
        balanced[~valid] = 0.0
        balanced = 0.5 * (balanced + balanced.T)

        balanced_total = float(balanced.sum())
        if np.isfinite(balanced_total) and balanced_total > 0.0:
            normalized = balanced / balanced_total

    # Adaptive exploration floor. Tiny graphs get a smaller absolute floor;
    # concentrated distributions get a stronger but still bounded floor.
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

    floor_strength = (
        0.006
        + (0.026 + 0.014 * graph_scale) * concentration_level
    )
    floor_strength = float(np.clip(floor_strength, 0.006, 0.045))
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
