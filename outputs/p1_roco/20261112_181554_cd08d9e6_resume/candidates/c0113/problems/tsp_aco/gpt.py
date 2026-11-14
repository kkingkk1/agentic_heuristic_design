import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    # Use only edges that are valid in both directions.
    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = np.full((n, n), np.inf, dtype=float)
    d[valid] = 0.5 * (distances[valid] + distances.T[valid])
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.max(finite_values)), 1.0)

    global_scale = max(global_scale, np.finfo(float).eps)

    penalty = max(float(np.max(finite_values)), global_scale) * 4.0
    penalty += global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    # Use several nearest neighbors for a more stable local scale.
    neighborhood_size = min(n - 1, max(3, int(np.ceil(np.sqrt(n)))))

    local_scale = np.empty(n, dtype=float)
    for i in range(n):
        row = work[i].copy()
        row[i] = np.inf
        nearest = np.sort(row)[:neighborhood_size]
        positive = nearest[np.isfinite(nearest) & (nearest > 0.0)]

        if positive.size:
            local_scale[i] = float(np.median(positive))
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(
        local_scale,
        global_scale * 1e-6 + np.finfo(float).eps,
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = work / pair_scale

    # Smooth distance preference avoids the overly sharp saturation of a
    # single exponential term.
    inverse_distance = 1.0 / (1.0 + relative_distance)
    exponential_distance = np.exp(-np.sqrt(np.clip(relative_distance, 0.0, 60.0)))
    distance_term = 0.60 * inverse_distance + 0.40 * exponential_distance

    rank_term = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    endpoint_quality = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 60.0)
    )
    endpoint_quality_t = np.exp(
        -np.clip(work / local_scale[None, :], 0.0, 60.0)
    )
    accessibility = np.sqrt(endpoint_quality * endpoint_quality_t)

    # Endpoint urgency: prioritize edges incident to nodes whose best
    # alternative is substantially worse.
    sorted_rows = np.sort(work, axis=1)
    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1]

    regret = (second - best) / np.maximum(second, global_scale * 1e-6)
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    mutual_nearest = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.22, 1.0)

    # Shared candidate-neighborhood preference.
    candidate_mask = valid & (ranks < neighborhood_size)
    candidate_counts = candidate_mask.sum(axis=1).astype(float)
    shared = candidate_mask.astype(float) @ candidate_mask.T.astype(float)
    denominator = np.sqrt(
        np.maximum(candidate_counts[:, None] * candidate_counts[None, :], 1.0)
    )
    shared_neighborhood = np.clip(shared / denominator, 0.0, 1.0)
    shared_bonus = 1.0 + 0.20 * shared_neighborhood

    # Softly suppress edges that are outliers relative to both endpoints.
    excess = np.maximum(relative_distance - 2.0, 0.0)
    outlier_penalty = 1.0 / (1.0 + 0.75 * excess)

    # Mild MST support retains globally useful connecting edges.
    mst_edges = np.zeros((n, n), dtype=bool)
    used = np.zeros(n, dtype=bool)
    min_cost = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    min_cost[0] = 0.0

    for _ in range(n):
        selectable = np.where(~used, min_cost, np.inf)
        node = int(np.argmin(selectable))

        if used[node] or not np.isfinite(selectable[node]):
            break

        used[node] = True

        if parent[node] >= 0:
            p = parent[node]
            if valid[p, node]:
                mst_edges[p, node] = True
                mst_edges[node, p] = True

        for neighbor in range(n):
            if not used[neighbor] and work[node, neighbor] < min_cost[neighbor]:
                min_cost[neighbor] = work[node, neighbor]
                parent[neighbor] = node

    mst_bonus = np.where(mst_edges, 1.12, 1.0)

    scores = (
        distance_term
        * (0.48 + 0.22 * rank_term + 0.30 * accessibility)
        * regret_bonus
        * mutual_bonus
        * shared_bonus
        * mst_bonus
        * outlier_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
