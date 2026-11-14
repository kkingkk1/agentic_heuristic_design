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

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return result

    raw = np.where(raw_valid, distances, np.inf)
    reverse = raw.T

    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)
    both_valid = forward_valid & backward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )

    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = reverse[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return result

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    row_order = np.argsort(distance, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    top_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)
    competitive_size = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)

    for i in range(n):
        candidates = row_order[i][valid[i, row_order[i]]]

        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        row_rank[i, candidates] = np.arange(
            candidates.size,
            dtype=float,
        )

        top_mask[i, candidates[:neighborhood_size]] = True

        if candidates.size > 1:
            second_distance[i] = values[1]
        else:
            second_distance[i] = values[0]

        positive_neighbors = values[values > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        : min(5, positive_neighbors.size)
                    ]
                )
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    safe_distance = np.where(valid, distance, np.inf)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_distance = safe_distance / np.maximum(
            pair_scale,
            eps,
        )
        adaptive_shortness = np.exp(
            -np.clip(relative_distance, 0.0, 50.0)
        )

        endpoint_shortness = np.sqrt(
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
    finite_ranks = np.isfinite(row_rank)

    rank_score[finite_ranks] = 1.0 / (
        1.0 + row_rank[finite_ranks]
    )

    reciprocal_rank = np.sqrt(
        rank_score * rank_score.T
    )

    nearest = np.argmin(distance, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest):
        if valid[i, j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    mutual_top = (
        top_mask
        & top_mask.T
        & valid
    ).astype(float)

    local_competition = (
        valid
        & (row_rank < competitive_size)
        & (row_rank.T < competitive_size)
    ).astype(float)

    shared_neighbors = (
        top_mask.astype(float)
        @ top_mask.astype(float).T
    )

    shared_neighbor_signal = np.clip(
        shared_neighbors / max(float(neighborhood_size), 1.0),
        0.0,
        1.0,
    )

    two_hop_bonus = (
        1.0
        + 0.08
        * shared_neighbor_signal
        * local_competition
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
        + 0.16
        * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    scarcity_ratio = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    ) / np.maximum(safe_distance, eps)

    scarcity_bonus = 1.0 + 0.12 * np.clip(
        scarcity_ratio - 1.0,
        0.0,
        4.0,
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

    structural_bonus = (
        1.0
        + 0.30 * mutual_nearest
        + 0.10 * mutual_top
    )

    scores = (
        adaptive_shortness
        * endpoint_shortness
        * (0.56 + 0.44 * reciprocal_rank)
        * structural_bonus
        * scarcity_bonus
        * opportunity_penalty
        * two_hop_bonus
        * isolation_bonus
    )

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
