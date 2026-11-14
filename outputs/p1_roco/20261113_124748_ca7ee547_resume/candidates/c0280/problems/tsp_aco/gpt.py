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

    diagonal = np.eye(n, dtype=bool)
    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return scores

    raw = np.where(raw_valid, distances, np.inf)

    # Use the shortest available directed representation, then enforce symmetry.
    safe = np.minimum(raw, raw.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return scores

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    top_neighbors = np.zeros((n, n), dtype=bool)

    neighborhood_size = min(
        max(3, int(np.ceil(np.sqrt(n)))),
        n - 1,
    )

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = row_order[i][valid[i, row_order[i]]]

        if candidates.size == 0:
            continue

        values = safe[i, candidates]

        row_rank[i, candidates] = np.arange(
            candidates.size,
            dtype=float,
        )

        selected = candidates[:neighborhood_size]
        top_neighbors[i, selected] = True
        nearest[i] = int(candidates[0])

        second_distance[i] = (
            values[1] if candidates.size > 1 else values[0]
        )

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
    local_scale = np.maximum(local_scale, global_scale * 0.04)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized_distance = safe / np.maximum(pair_scale, eps)

        distance_kernel = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0) ** 1.15
        )

        endpoint_kernel = np.sqrt(
            np.exp(
                -np.clip(
                    safe / np.maximum(local_scale[:, None], eps),
                    0.0,
                    50.0,
                )
            )
            * np.exp(
                -np.clip(
                    safe / np.maximum(local_scale[None, :], eps),
                    0.0,
                    50.0,
                )
            )
        )

    row_rank_score = np.zeros((n, n), dtype=float)
    finite_rank = row_rank < n
    row_rank_score[finite_rank] = 1.0 / (
        1.0 + row_rank[finite_rank]
    )

    reciprocal_rank = np.sqrt(
        row_rank_score * row_rank_score.T
    )

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )

    neighborhood_sizes = top_neighbors.sum(axis=1).astype(float)
    union_sizes = (
        neighborhood_sizes[:, None]
        + neighborhood_sizes[None, :]
        - shared_neighbors
    )

    overlap = shared_neighbors / np.maximum(union_sizes, 1.0)
    locally_competitive = (
        valid
        & (row_rank < neighborhood_size)
        & (row_rank.T < neighborhood_size)
    ).astype(float)

    complementary_coverage = 1.0 + (
        0.14
        * (1.0 - np.clip(overlap, 0.0, 1.0))
        * locally_competitive
    )

    shared_neighbor_bonus = 1.0 + (
        0.08
        * np.clip(
            shared_neighbors / max(float(neighborhood_size), 1.0),
            0.0,
            1.0,
        )
        * locally_competitive
    )

    scarcity_ratio = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    ) / np.maximum(safe, eps)

    scarcity_bonus = 1.0 + 0.14 * np.clip(
        scarcity_ratio - 1.0,
        0.0,
        4.0,
    )

    popularity = top_neighbors.astype(float).sum(axis=0)
    popularity_reference = max(float(np.mean(popularity)), 1.0)
    popularity_ratio = np.clip(
        popularity / popularity_reference,
        0.0,
        3.0,
    )

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + 0.12 * popularity_ratio[:, None])
        * (1.0 + 0.12 * popularity_ratio[None, :])
    )

    mutual_bonus = (
        1.0
        + 0.30 * mutual_nearest
        + 0.12 * mutual_top
    )

    scores = (
        distance_kernel
        * endpoint_kernel
        * (0.48 + 0.52 * reciprocal_rank)
        * mutual_bonus
        * complementary_coverage
        * shared_neighbor_bonus
        * scarcity_bonus
        * concentration_penalty
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
