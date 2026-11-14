import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return result

    forward = np.where(directed_valid, distances, np.inf)
    backward = forward.T

    valid = np.isfinite(forward) | np.isfinite(backward)
    safe = np.minimum(forward, backward)
    safe[~valid] = np.inf
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe) & ~diagonal
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    neighbor_count = min(5, n - 1)
    nearest_distances = np.take_along_axis(
        safe,
        order[:, :neighbor_count],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_distances)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_distances, 0.0),
        axis=1,
    ) / np.maximum(nearest_valid.sum(axis=1), 1)

    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        safe,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    distance_kernel = np.exp(-1.15 * relative_distance)

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

    rank_agreement = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )

    top_k = min(3, n - 1)
    top_neighbors = (
        (row_rank < top_k)
        & valid
    )

    nearest = order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    mutual_top = (
        top_neighbors
        & top_neighbors.T
    ).astype(float)

    local_competition = (
        valid
        & (row_rank < top_k)
        & (row_rank.T < top_k)
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
    complementary_coverage = (
        1.0
        + 0.14
        * (1.0 - np.clip(overlap, 0.0, 1.0))
        * local_competition
    )

    candidate_popularity = top_neighbors.astype(float).sum(axis=0)
    popularity_reference = max(
        float(np.mean(candidate_popularity)),
        1.0,
    )
    popularity_ratio = np.clip(
        candidate_popularity / popularity_reference,
        0.0,
        3.0,
    )

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + 0.12 * popularity_ratio[:, None])
        * (1.0 + 0.12 * popularity_ratio[None, :])
    )

    valid_counts = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    isolation_ratio = np.divide(
        pair_scale,
        safe + global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    isolation_bonus = 1.0 + 0.10 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    mutual_bonus = (
        1.0
        + 0.26 * mutual_nearest
        + 0.10 * mutual_top
    )

    scores = (
        distance_kernel
        * endpoint_kernel
        * (0.54 + 0.46 * rank_agreement)
        * mutual_bonus
        * (0.90 + 0.20 * scarcity_pair)
        * complementary_coverage
        * concentration_penalty
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
