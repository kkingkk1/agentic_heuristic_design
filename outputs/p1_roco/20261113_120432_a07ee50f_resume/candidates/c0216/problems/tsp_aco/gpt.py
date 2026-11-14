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

    finite_values = distances[directed_valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    directed = np.where(directed_valid, distances, np.inf)
    reverse = directed.T

    valid = np.isfinite(directed) | np.isfinite(reverse)
    safe = np.minimum(directed, reverse)
    safe[~valid] = np.inf
    np.fill_diagonal(safe, np.inf)
    valid &= np.isfinite(safe)

    if not np.any(valid):
        return result

    valid_counts = valid.sum(axis=1).astype(float)

    neighborhood_size = min(5, n - 1)
    nearest_values = np.partition(
        safe,
        kth=neighborhood_size - 1,
        axis=1,
    )[:, :neighborhood_size]

    finite_neighborhood = np.isfinite(nearest_values)
    neighborhood_count = finite_neighborhood.sum(axis=1)

    local_scale = np.sum(
        np.where(finite_neighborhood, nearest_values, 0.0),
        axis=1,
    ) / np.maximum(neighborhood_count, 1)

    local_scale = np.where(
        neighborhood_count > 0,
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
        where=np.isfinite(safe),
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    distance_kernel = np.exp(-1.15 * relative_distance)

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

    rank_window = min(8, n - 1)
    candidates = np.argpartition(
        safe,
        kth=rank_window - 1,
        axis=1,
    )[:, :rank_window]

    candidate_distances = np.take_along_axis(
        safe,
        candidates,
        axis=1,
    )
    candidate_order = np.argsort(
        candidate_distances,
        axis=1,
        kind="stable",
    )
    neighbors = np.take_along_axis(
        candidates,
        candidate_order,
        axis=1,
    )

    row_rank = np.full((n, n), float(n), dtype=float)
    np.put_along_axis(
        row_rank,
        neighbors,
        np.broadcast_to(
            np.arange(rank_window, dtype=float),
            (n, rank_window),
        ),
        axis=1,
    )

    col_rank = row_rank.T
    rank_agreement = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    nearest = neighbors[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    np.put_along_axis(
        top_neighbors,
        neighbors[:, :top_k],
        True,
        axis=1,
    )

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    local_competition = (
        (row_rank < top_k)
        & (col_rank < top_k)
        & valid
    ).astype(float)

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = np.divide(
        safe,
        np.maximum(endpoint_best, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=np.isfinite(safe),
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.14 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    isolation_bonus = 1.0 + 0.18 * np.clip(
        pair_scale / (safe + global_scale),
        0.0,
        1.0,
    )

    scores = (
        distance_kernel
        * endpoint_kernel
        * (0.52 + 0.48 * rank_agreement)
        * (1.0 + 0.26 * mutual_nearest + 0.08 * mutual_top)
        * (1.0 + 0.08 * local_competition)
        * opportunity_penalty
        * (0.88 + 0.24 * scarcity_pair)
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
