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

    directed = np.where(directed_valid, distances, np.inf)
    reverse = directed.T

    valid = np.isfinite(directed) | np.isfinite(reverse)
    safe = np.minimum(directed, reverse)
    safe[~valid] = np.inf
    np.fill_diagonal(safe, np.inf)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = safe[np.isfinite(safe)]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    valid_counts = valid.sum(axis=1).astype(float)
    k_local = min(5, n - 1)
    k_rank = min(8, n - 1)
    k_core = min(3, n - 1)

    nearest_values = np.partition(
        safe,
        kth=k_local - 1,
        axis=1,
    )[:, :k_local]

    finite_nearest = np.isfinite(nearest_values)
    nearest_count = finite_nearest.sum(axis=1)

    local_scale = np.sum(
        np.where(finite_nearest, nearest_values, 0.0),
        axis=1,
    ) / np.maximum(nearest_count, 1)

    local_scale = np.where(
        nearest_count > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.05 * global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        safe,
        pair_scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    distance_score = np.exp(-1.05 * relative_distance)

    candidates = np.argpartition(
        safe,
        kth=k_rank - 1,
        axis=1,
    )[:, :k_rank]

    candidate_distances = np.take_along_axis(
        safe,
        candidates,
        axis=1,
    )

    order = np.argsort(
        candidate_distances,
        axis=1,
        kind="stable",
    )

    neighbors = np.take_along_axis(candidates, order, axis=1)
    ordered_distances = np.take_along_axis(
        candidate_distances,
        order,
        axis=1,
    )

    row_rank = np.full((n, n), float(n), dtype=float)
    np.put_along_axis(
        row_rank,
        neighbors,
        np.broadcast_to(
            np.arange(k_rank, dtype=float),
            (n, k_rank),
        ),
        axis=1,
    )

    col_rank = row_rank.T
    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    first_distance = ordered_distances[:, 0]
    if k_rank > 1:
        second_distance = ordered_distances[:, 1]
    else:
        second_distance = first_distance

    first_distance = np.where(
        np.isfinite(first_distance),
        first_distance,
        global_scale,
    )
    second_distance = np.where(
        np.isfinite(second_distance),
        second_distance,
        first_distance + global_scale,
    )

    regret_scale = np.maximum(second_distance, global_scale * 0.05)
    endpoint_regret = np.clip(
        (second_distance[:, None] - safe)
        / regret_scale[:, None],
        0.0,
        1.0,
    )
    endpoint_regret += np.clip(
        (second_distance[None, :] - safe)
        / regret_scale[None, :],
        0.0,
        1.0,
    )
    endpoint_regret *= 0.5

    nearest = neighbors[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_neighbors = np.zeros((n, n), dtype=bool)
    np.put_along_axis(
        top_neighbors,
        neighbors[:, :k_core],
        True,
        axis=1,
    )

    mutual_core = (
        valid
        & top_neighbors
        & top_neighbors.T
    ).astype(float)

    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    opportunity = np.divide(
        safe,
        np.minimum(local_scale[:, None], local_scale[None, :]),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    opportunity_penalty = 1.0 / (
        1.0 + 0.16 * np.clip(opportunity - 1.0, 0.0, 6.0)
    )

    scores = (
        distance_score
        * (0.48 + 0.52 * rank_score)
        * (0.82 + 0.36 * endpoint_regret)
        * (1.0 + 0.24 * mutual_nearest + 0.10 * mutual_core)
        * (0.86 + 0.28 * scarcity_pair)
        * opportunity_penalty
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
