import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    eps = np.finfo(float).eps
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

    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return result

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)

    local_k = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)
    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    top_neighbors = np.zeros((n, n), dtype=bool)

    for i in range(n):
        neighbors = row_order[i][valid[i, row_order[i]]]

        if neighbors.size == 0:
            continue

        values = safe[i, neighbors]

        row_rank[i, neighbors] = np.arange(
            neighbors.size,
            dtype=float,
        )

        top_neighbors[i, neighbors[:local_k]] = True

        if values.size > 1:
            second_distance[i] = values[1]
        else:
            second_distance[i] = values[0]

        local_values = values[: min(local_k, values.size)]
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
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = safe / np.maximum(pair_scale, eps)
        distance_score = np.exp(
            -np.clip(relative_distance, 0.0, 50.0)
        )

        endpoint_score = np.sqrt(
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
    finite_rank = np.isfinite(row_rank)
    rank_score[finite_rank] = 1.0 / (
        1.0 + row_rank[finite_rank]
    )

    reciprocal_rank = np.sqrt(
        rank_score * rank_score.T
    )

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        neighbors = row_order[i][valid[i, row_order[i]]]
        if neighbors.size:
            nearest[i] = neighbors[0]

    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    mutual_local = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    union_size = (
        top_neighbors.sum(axis=1)[:, None]
        + top_neighbors.sum(axis=1)[None, :]
        - shared_neighbors
    )

    overlap = shared_neighbors / np.maximum(union_size, 1.0)

    competitive = (
        valid
        & (row_rank < local_k)
        & (row_rank.T < local_k)
    ).astype(float)

    complementary_bonus = (
        1.0
        + 0.12
        * (1.0 - np.clip(overlap, 0.0, 1.0))
        * competitive
    )

    valid_counts = valid.sum(axis=1).astype(float)
    node_scarcity = 1.0 / np.sqrt(
        np.maximum(valid_counts, 1.0)
    )

    scarcity_pair = 0.5 * (
        node_scarcity[:, None]
        + node_scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    opportunity_ratio = safe / np.maximum(
        endpoint_best,
        eps,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.14
        * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    isolation_bonus = 1.0 + 0.10 * np.clip(
        pair_scale / (safe + global_scale),
        0.0,
        1.0,
    )

    structural_bonus = (
        1.0
        + 0.30 * mutual_nearest
        + 0.10 * mutual_local
    )

    scores = (
        distance_score
        * endpoint_score
        * (0.52 + 0.48 * reciprocal_rank)
        * structural_bonus
        * complementary_bonus
        * opportunity_penalty
        * (0.90 + 0.20 * scarcity_pair)
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

    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
