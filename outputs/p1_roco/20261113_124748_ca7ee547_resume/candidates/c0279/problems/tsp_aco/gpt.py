import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return scores

    raw = np.where(raw_valid, distances, np.inf)

    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(raw.T)
    both_valid = forward_valid & backward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        raw[both_valid] + raw.T[both_valid]
    )

    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = raw.T[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    row_order = np.argsort(distance, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.25 * np.sqrt(n)))),
    )
    competitive_size = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n)))),
    )

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    for i in range(n):
        candidates = row_order[i][valid[i, row_order[i]]]

        if candidates.size == 0:
            continue

        values = distance[i, candidates]

        row_rank[i, candidates] = np.arange(
            candidates.size,
            dtype=float,
        )

        top_candidates = candidates[:neighborhood_size]
        top_mask[i, top_candidates] = True

        second_distance[i] = (
            values[1] if candidates.size > 1 else values[0]
        )

        positive_neighbors = values[values > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        :min(6, positive_neighbors.size)
                    ]
                )
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(
        np.where(np.isfinite(local_scale), local_scale, global_scale),
        global_scale * 0.04,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    safe_distance = np.where(valid, distance, np.inf)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized_distance = safe_distance / np.maximum(
            pair_scale,
            eps,
        )

        distance_score = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0)
        )

        endpoint_affinity = np.sqrt(
            np.exp(
                -np.clip(
                    safe_distance / local_scale[:, None],
                    0.0,
                    50.0,
                )
            )
            * np.exp(
                -np.clip(
                    safe_distance / local_scale[None, :],
                    0.0,
                    50.0,
                )
            )
        )

    rank_score = np.zeros((n, n), dtype=float)
    finite_rank = np.isfinite(row_rank)
    rank_score[finite_rank] = 1.0 / (
        1.0 + row_rank[finite_rank]
    )

    reciprocal_rank = np.sqrt(
        rank_score * rank_score.T
    )

    mutual_nearest = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = row_order[i][valid[i, row_order[i]]]
        if candidates.size:
            nearest[i] = int(candidates[0])

    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    mutual_top = (
        top_mask
        & top_mask.T
        & valid
    ).astype(float)

    locally_competitive = (
        valid
        & (row_rank < float(competitive_size))
        & (row_rank.T < float(competitive_size))
    ).astype(float)

    shared_neighbors = (
        top_mask.astype(float)
        @ top_mask.astype(float).T
    )

    top_counts = top_mask.sum(axis=1).astype(float)
    union_counts = (
        top_counts[:, None]
        + top_counts[None, :]
        - shared_neighbors
    )

    neighborhood_overlap = shared_neighbors / np.maximum(
        union_counts,
        1.0,
    )

    complementary_coverage = (
        1.0
        + 0.12
        * np.clip(1.0 - neighborhood_overlap, 0.0, 1.0)
        * locally_competitive
    )

    two_hop_bonus = (
        1.0
        + 0.07
        * np.clip(
            shared_neighbors / max(float(neighborhood_size), 1.0),
            0.0,
            1.0,
        )
        * locally_competitive
    )

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe_distance / np.maximum(
        endpoint_best,
        eps,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.15
        * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    scarcity = np.ones((n, n), dtype=float)

    for i in range(n):
        row_mask = valid[i]
        if not np.any(row_mask):
            continue

        gap = max(second_distance[i], global_scale)
        ratio = gap / np.maximum(distance[i, row_mask], eps)

        scarcity[i, row_mask] += 0.32 * np.clip(
            ratio - 1.0,
            0.0,
            5.0,
        )

    scarcity = np.sqrt(scarcity * scarcity.T)

    local_strength = np.where(
        valid,
        np.exp(
            -np.clip(
                safe_distance / local_scale[:, None],
                0.0,
                50.0,
            )
        )
        / np.sqrt(1.0 + np.maximum(row_rank, 0.0)),
        0.0,
    )

    strongest_strength = np.max(local_strength, axis=1)
    total_strength = np.sum(local_strength, axis=1)

    concentration = strongest_strength / np.maximum(
        total_strength,
        eps,
    )
    concentration = np.clip(concentration, 0.0, 1.0)

    sorted_strength = np.sort(local_strength, axis=1)
    second_strength = sorted_strength[:, -2]

    alternative_ratio = second_strength / np.maximum(
        strongest_strength,
        eps,
    )
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)

    sparsity = local_scale / np.maximum(global_scale, eps)
    sparsity = np.clip(sparsity, 0.0, 2.0)
    sparsity = sparsity / (1.0 + sparsity)

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + row_rank.T), 0.0, 1.0)
    )

    concentration_penalty = (
        np.sqrt(concentration[:, None] * concentration[None, :])
        * edge_topness
    )

    complementary_opportunity = np.sqrt(
        alternative_ratio[:, None] * alternative_ratio[None, :]
    )

    sparse_alternative_bonus = (
        np.sqrt(sparsity[:, None] * sparsity[None, :])
        * (1.0 - edge_topness)
    )

    mutual_bonus = (
        1.0
        + 0.26 * mutual_nearest
        + 0.09 * mutual_top
    )

    isolation_ratio = pair_scale / (
        safe_distance + 0.85 * global_scale
    )

    isolation_bonus = 1.0 + 0.12 * np.clip(
        np.nan_to_num(
            isolation_ratio,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    base_scores = (
        distance_score
        * endpoint_affinity
        * (0.57 + 0.43 * reciprocal_rank)
        * mutual_bonus
        * opportunity_penalty
        * two_hop_bonus
        * complementary_coverage
        * scarcity
        * isolation_bonus
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_scores = np.log(base_scores)

    log_scores += (
        0.06
        * complementary_opportunity
        * locally_competitive
    )

    log_scores += 0.045 * sparse_alternative_bonus
    log_scores -= 0.07 * concentration_penalty

    scores = np.exp(np.clip(log_scores, -60.0, 30.0))
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
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
