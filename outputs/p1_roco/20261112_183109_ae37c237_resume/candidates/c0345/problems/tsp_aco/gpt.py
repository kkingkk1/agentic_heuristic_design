import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""

    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 8.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]

    if n > 2:
        second = sorted_distances[:, 1]
        third = sorted_distances[:, 2]
    else:
        second = best
        third = second

    valid_count = np.sum(valid, axis=1).astype(float)

    local_scale = np.where(
        valid_count > 0.0,
        np.where(
            np.isfinite(second) & (second > 0.0),
            0.5 * (best + second),
            global_scale,
        ),
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=pair_scale > 0.0,
    )

    relative_distance = np.nan_to_num(
        relative_distance,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )

    rank_scale = max(1.0, 0.72 * np.sqrt(float(max(n, 2))))
    rank_term = np.exp(
        -np.minimum(ranks, 60.0) / rank_scale
    )
    bidirectional_rank = np.sqrt(
        rank_term * rank_term.T
    )

    short_edge_term = np.exp(
        -np.clip(1.28 * relative_distance, 0.0, 60.0)
    )

    endpoint_term = np.exp(
        -np.clip(
            0.52
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(
        np.nan_to_num(regret, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )

    next_gap = np.divide(
        third - second,
        np.maximum(third, eps),
    )
    next_gap = np.clip(
        np.nan_to_num(next_gap, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )

    regret_term = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )
    regret_term *= 1.0 + 0.10 * (
        next_gap[:, None] + next_gap[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_term = np.where(mutual_nearest, 1.34, 1.0)

    candidate_reference = np.median(
        valid_count[valid_count > 0.0]
    )
    if not np.isfinite(candidate_reference) or candidate_reference <= 0.0:
        candidate_reference = 1.0

    scarcity = np.sqrt(
        candidate_reference / np.maximum(valid_count, 1.0)
    )
    scarcity = np.clip(scarcity, 0.84, 1.22)

    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    scarcity_term = np.clip(scarcity_term, 0.90, 1.14)

    close_i = (
        work <= 1.45 * local_scale[:, None]
    )
    close_j = (
        work <= 1.45 * local_scale[None, :]
    )

    close_i[~valid] = False
    close_j[~valid] = False

    local_degree = (
        np.sum(close_i, axis=1).astype(float)
        + np.sum(close_j, axis=0).astype(float)
    )

    degree_reference_values = local_degree[local_degree > 0.0]
    degree_reference = (
        float(np.median(degree_reference_values))
        if degree_reference_values.size
        else 1.0
    )

    endpoint_pressure = np.sqrt(
        degree_reference / np.maximum(local_degree, 1.0)
    )
    endpoint_pressure = np.clip(endpoint_pressure, 0.90, 1.16)

    feasibility_term = np.sqrt(
        endpoint_pressure[:, None] * endpoint_pressure[None, :]
    )
    feasibility_term = np.clip(feasibility_term, 0.92, 1.10)

    shared_neighbors = (
        close_i.astype(float) @ close_j.astype(float)
    )

    degree_sum = (
        local_degree[:, None] + local_degree[None, :]
    )

    overlap = np.divide(
        2.0 * shared_neighbors,
        np.maximum(degree_sum, 1.0),
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_term = 1.0 + 0.045 * (1.0 - overlap)
    bridge_term = np.clip(bridge_term, 1.0, 1.045)

    long_edge_penalty = np.exp(
        -0.92 * np.maximum(relative_distance - 1.0, 0.0)
    )

    valid_count_pair = np.minimum(
        valid_count[:, None],
        valid_count[None, :],
    )
    sparse_edge_guard = np.where(
        valid_count_pair <= 1.0,
        1.08,
        1.0,
    )

    base = (
        0.49 * np.log(np.maximum(short_edge_term, eps))
        + 0.28 * np.log(np.maximum(bidirectional_rank, eps))
        + 0.15 * np.log(np.maximum(endpoint_term, eps))
        + 0.08 * np.log(np.maximum(long_edge_penalty, eps))
    )

    score_log = (
        base
        + np.log(np.maximum(regret_term, eps))
        + np.log(np.maximum(mutual_term, eps))
        + np.log(np.maximum(scarcity_term, eps))
        + np.log(np.maximum(feasibility_term, eps))
        + np.log(np.maximum(bridge_term, eps))
        + np.log(np.maximum(sparse_edge_guard, eps))
    )

    scores = np.exp(np.clip(score_log, -60.0, 20.0))
    scores[~valid] = 0.0

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
