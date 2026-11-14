import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
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

    symmetric = np.full((n, n), np.inf, dtype=float)
    both = forward_valid & backward_valid
    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid

    symmetric[both] = 0.5 * (raw[both] + raw.T[both])
    symmetric[only_forward] = raw[only_forward]
    symmetric[only_backward] = raw.T[only_backward]

    np.fill_diagonal(symmetric, np.inf)
    valid = np.isfinite(symmetric) & ~diagonal

    if not np.any(valid):
        return scores

    finite_values = symmetric[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(symmetric, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    top_k = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    top_neighbors = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        values = symmetric[i, candidates]
        ranks[i, candidates] = np.arange(candidates.size, dtype=float)
        top_neighbors[i, candidates[:top_k]] = True

        second_distance[i] = (
            values[1] if values.size > 1 else values[0]
        )

        positive = values[values > 0.0]
        if positive.size:
            local_scale[i] = float(
                np.median(positive[: min(5, positive.size)])
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(
        np.where(np.isfinite(local_scale), local_scale, global_scale),
        eps,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    safe_distance = np.where(valid, symmetric, np.inf)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized_distance = safe_distance / np.maximum(
            pair_scale,
            eps,
        )

        distance_kernel = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0)
        )

        endpoint_kernel = np.sqrt(
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

    rank_kernel = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    nearest = np.argmin(safe_distance, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest):
        if valid[i, j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    competitive_k = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)
    competitive = (
        valid
        & (ranks < competitive_k)
        & (ranks.T < competitive_k)
    ).astype(float)

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    two_hop_bonus = 1.0 + 0.08 * competitive * np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
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
        + 0.14 * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    scarcity_ratio = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    ) / np.maximum(safe_distance, eps)

    scarcity_bonus = 1.0 + 0.10 * np.clip(
        scarcity_ratio - 1.0,
        0.0,
        4.0,
    )

    local_strength = np.where(
        valid,
        np.exp(
            -np.clip(
                safe_distance / local_scale[:, None],
                0.0,
                50.0,
            )
        ) / np.sqrt(1.0 + ranks),
        0.0,
    )

    strongest = np.max(local_strength, axis=1)
    total = np.sum(local_strength, axis=1)

    concentration = strongest / np.maximum(total, eps)
    concentration = np.clip(concentration, 0.0, 1.0)

    sorted_strength = np.sort(local_strength, axis=1)
    second_strength = sorted_strength[:, -2]
    alternatives = second_strength / np.maximum(strongest, eps)
    alternatives = np.clip(alternatives, 0.0, 1.0)

    edge_topness = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    concentration_penalty = (
        np.sqrt(
            concentration[:, None]
            * concentration[None, :]
        )
        * edge_topness
    )

    complementary_opportunity = np.sqrt(
        alternatives[:, None]
        * alternatives[None, :]
    ) * (1.0 - edge_topness)

    structural_bonus = (
        1.0
        + 0.28 * mutual_nearest
        + 0.10 * mutual_top
    )

    base_scores = (
        distance_kernel
        * endpoint_kernel
        * (0.56 + 0.44 * rank_kernel)
        * structural_bonus
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
        * competitive
    )
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
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
