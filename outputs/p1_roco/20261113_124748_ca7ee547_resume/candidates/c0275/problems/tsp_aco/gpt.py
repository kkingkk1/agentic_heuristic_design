import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized prior for stochastic TSP edge sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return result

    raw = np.where(directed_valid, distances, np.inf)
    reverse = raw.T

    valid = np.isfinite(raw) | np.isfinite(reverse)
    safe = np.minimum(raw, reverse)
    safe[~valid] = np.inf
    np.fill_diagonal(safe, np.inf)
    valid &= np.isfinite(safe)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    np.put_along_axis(
        row_rank,
        order,
        np.broadcast_to(
            np.arange(n, dtype=float),
            (n, n),
        ),
        axis=1,
    )

    valid_counts = valid.sum(axis=1).astype(float)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n)))),
    )
    top_k = min(n - 1, max(2, int(np.ceil(0.75 * np.sqrt(n)))))

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        values = safe[i, candidates]

        if candidates.size > 1:
            second_distance[i] = values[1]
        else:
            second_distance[i] = values[0]

        local_values = values[:min(neighborhood_size, values.size)]
        positive_local = local_values[local_values > 0.0]

        if positive_local.size:
            local_scale[i] = float(np.median(positive_local))
        elif local_values.size:
            local_scale[i] = float(np.median(local_values))

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_distance = safe / np.maximum(pair_scale, eps)
        distance_kernel = np.exp(
            -1.10 * np.clip(relative_distance, 0.0, 50.0)
        )

        endpoint_kernel = np.sqrt(
            np.exp(
                -np.clip(
                    safe / local_scale[:, None],
                    0.0,
                    50.0,
                )
            )
            * np.exp(
                -np.clip(
                    safe / local_scale[None, :],
                    0.0,
                    50.0,
                )
            )
        )

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + row_rank[valid])

    reciprocal_rank = np.sqrt(rank_score * rank_score.T)

    top_neighbors = (
        valid
        & (row_rank < float(top_k))
    )

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    nearest = order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    local_competition = (
        valid
        & (row_rank < float(top_k))
        & (row_rank.T < float(top_k))
    ).astype(float)

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe / np.maximum(endpoint_best, eps)
    opportunity_penalty = 1.0 / (
        1.0
        + 0.15 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    scarcity_ratio = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    ) / np.maximum(safe, eps)

    scarcity_bonus = 1.0 + 0.14 * np.clip(
        scarcity_ratio - 1.0,
        0.0,
        5.0,
    )

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    shared_neighbor_signal = np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
    )

    two_hop_bonus = (
        1.0
        + 0.08
        * shared_neighbor_signal
        * local_competition
    )

    local_strength = np.where(
        valid,
        np.exp(
            -np.clip(
                safe / local_scale[:, None],
                0.0,
                50.0,
            )
        )
        / np.sqrt(1.0 + row_rank),
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

    sparsity = np.clip(
        local_scale / np.maximum(global_scale, eps),
        0.0,
        3.0,
    )
    sparsity = sparsity / (1.0 + sparsity)

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + row_rank.T), 0.0, 1.0)
    )

    concentration_penalty = (
        np.sqrt(
            concentration[:, None]
            * concentration[None, :]
        )
        * edge_topness
    )

    complementary_opportunity = np.sqrt(
        alternative_ratio[:, None]
        * alternative_ratio[None, :]
    )

    sparse_alternative_bonus = (
        np.sqrt(
            sparsity[:, None]
            * sparsity[None, :]
        )
        * (1.0 - edge_topness)
    )

    mutual_bonus = (
        1.0
        + 0.28 * mutual_nearest
        + 0.10 * mutual_top
    )

    base_scores = (
        distance_kernel
        * endpoint_kernel
        * (0.56 + 0.44 * reciprocal_rank)
        * mutual_bonus
        * opportunity_penalty
        * scarcity_bonus
        * two_hop_bonus
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_scores = np.log(base_scores)

    log_scores += (
        0.06
        * complementary_opportunity
        * local_competition
    )
    log_scores += 0.045 * sparse_alternative_bonus
    log_scores -= 0.065 * concentration_penalty

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
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
