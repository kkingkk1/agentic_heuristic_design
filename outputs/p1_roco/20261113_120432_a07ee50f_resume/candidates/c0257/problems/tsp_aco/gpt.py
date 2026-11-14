import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    raw_valid = finite_nonnegative & ~diagonal

    if not np.any(raw_valid):
        return np.zeros((n, n), dtype=float)

    # Build a symmetric edge view. An edge is valid when either direction is valid.
    safe = np.full((n, n), np.inf, dtype=float)
    safe[raw_valid] = distances[raw_valid]

    reverse = safe.T
    safe = np.minimum(safe, reverse)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Per-node neighborhood scales use only finite neighbors.
    usable = min(5, n - 1)
    local_scale = np.empty(n, dtype=float)

    for i in range(n):
        values = safe[i, valid[i]]
        values = np.sort(values)
        nearest = values[:usable]

        positive_nearest = nearest[nearest > 0.0]
        if positive_nearest.size:
            scale = float(np.median(positive_nearest))
        else:
            scale = global_scale

        local_scale[i] = max(scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    distance_quality = np.exp(-np.clip(normalized_distance, 0.0, 30.0))
    distance_quality[~valid] = 0.0

    # Stable symmetric rank signal.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    row_rank[np.arange(n)[:, None], row_order[:, : n - 1]] = np.arange(n - 1)
    col_rank[col_order[: n - 1, :], np.arange(n)[None, :]] = np.arange(n - 1)[:, None]

    rank_quality = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    rank_quality[~valid] = 0.0

    # Top-neighborhood structure for mutuality and complementarity.
    top_k = min(4, n - 1)
    top_mask = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = row_order[i]
        candidates = candidates[valid[i, candidates]][:top_k]
        top_mask[i, candidates] = 1.0

    mutual_top = top_mask * top_mask.T
    mutual_top[~valid] = 0.0

    # Reward edges joining nodes with complementary, rather than redundant,
    # local neighborhoods.
    shared_neighbors = top_mask @ top_mask.T
    neighborhood_overlap = shared_neighbors / max(float(top_k), 1.0)
    complementarity = np.clip(1.0 - neighborhood_overlap, 0.0, 1.0)
    complementarity[~valid] = 0.0

    # Endpoint-load surrogate: avoid concentrating probability on nodes that
    # already have many similarly attractive incident edges.
    local_attractiveness = distance_quality * (0.55 + 0.45 * rank_quality)
    endpoint_load = np.sum(local_attractiveness, axis=1)
    load_scale = float(np.median(endpoint_load[endpoint_load > 0.0]))
    load_scale = max(load_scale, eps)

    normalized_load = np.clip(endpoint_load / load_scale, 0.0, 4.0)
    load_balance = 1.0 / np.sqrt(
        (1.0 + 0.22 * normalized_load[:, None])
        * (1.0 + 0.22 * normalized_load[None, :])
    )
    load_balance = 0.82 + 0.18 * load_balance
    load_balance[~valid] = 0.0

    # Bounded two-hop completion signal. Only a small nearest-neighbor set is
    # used to avoid an unbounded multi-hop or global-path bias.
    completion = np.zeros((n, n), dtype=float)
    bridge_k = min(4, n - 1)

    for i in range(n):
        intermediates = row_order[i]
        intermediates = intermediates[valid[i, intermediates]][:bridge_k]

        for j in range(n):
            if not valid[i, j]:
                continue

            candidates = intermediates[intermediates != j]
            if candidates.size == 0:
                completion[i, j] = 0.0
                continue

            leg_1 = safe[i, candidates]
            leg_2 = safe[candidates, j]
            feasible = np.isfinite(leg_1) & np.isfinite(leg_2)

            if np.any(feasible):
                best_two_hop = float(
                    np.min(leg_1[feasible] + leg_2[feasible])
                )
                completion[i, j] = np.exp(
                    -np.clip(best_two_hop / (2.0 * global_scale), 0.0, 20.0)
                )

    completion = 0.5 * (completion + completion.T)
    completion[~valid] = 0.0

    # Keep raw distance quality dominant; add structural terms with bounded
    # influence so diversity does not overwhelm cost ordering.
    structural = (
        0.52 * rank_quality
        + 0.16 * mutual_top
        + 0.14 * complementarity
        + 0.10 * completion
        + 0.08 * load_balance
    )

    scores = distance_quality * (0.58 + 0.42 * structural)
    scores *= load_balance

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
