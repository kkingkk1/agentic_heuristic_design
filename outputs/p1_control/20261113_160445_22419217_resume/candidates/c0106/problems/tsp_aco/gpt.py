import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if np.any(np.isnan(distances)) or np.any(distances < 0.0):
        raise ValueError("distance_matrix must contain nonnegative values")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    valid = np.isfinite(distances)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, distances, np.inf)

    positive = distances[valid & (distances > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    # Robust endpoint-local scales from the closest few neighbors.
    neighbor_count = min(3, n - 1)
    sorted_distances = np.sort(work, axis=1)
    local_neighbors = sorted_distances[:, :neighbor_count]

    local_scale = np.median(
        np.where(np.isfinite(local_neighbors), local_neighbors, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    # Local Gibbs affinity, with a weak global-scale component.
    local_affinity = np.exp(
        -np.minimum(np.where(valid, distances, 0.0) / pair_scale, 60.0)
    )
    global_affinity = np.exp(
        -np.minimum(np.where(valid, distances, 0.0) / global_scale, 60.0)
    )
    affinity = np.sqrt(local_affinity * global_affinity)
    affinity[~valid] = 0.0

    # Stable endpoint ranks and bidirectional rank agreement.
    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_decay = np.exp(-ranks / max(2.0, np.sqrt(float(n))))
    rank_decay[~valid] = 0.0
    reciprocal_rank = np.sqrt(rank_decay * rank_decay.T)

    # Regret: edges become important when the best local alternative is weak.
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, min(1, n - 2)]

    nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    second = np.where(np.isfinite(second), second, nearest)

    regret = np.maximum(second - nearest, 0.0)
    regret_scale = regret[regret > 0.0]
    regret_scale = (
        float(np.median(regret_scale))
        if regret_scale.size
        else global_scale
    )
    regret_scale = max(regret_scale, np.finfo(float).eps)

    endpoint_regret = np.tanh(regret / regret_scale)
    regret_pair = np.sqrt(
        endpoint_regret[:, None] * endpoint_regret[None, :]
    )

    # Mutual nearest and mutual top-three-neighbor structure.
    nearest_index = np.argmin(work, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    rows = np.arange(n)
    mutual_flags = nearest_index[nearest_index] == rows
    mutual_nearest[rows[mutual_flags], nearest_index[mutual_flags]] = 1.0
    mutual_nearest[
        nearest_index[mutual_flags], rows[mutual_flags]
    ] = 1.0

    top_k = ranks < float(neighbor_count)
    mutual_top_k = (top_k & top_k.T).astype(float)

    # Prim-style bridge prior retains useful inter-cluster edges.
    bridge = np.zeros((n, n), dtype=float)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        node = int(np.argmin(candidates))
        if not np.isfinite(candidates[node]):
            break

        in_tree[node] = True
        if parent[node] >= 0:
            other = parent[node]
            bridge[node, other] = 1.0
            bridge[other, node] = 1.0

        update = (~in_tree) & valid[node] & (work[node] < best)
        best[update] = work[node, update]
        parent[update] = node

    # Combine complementary signals while keeping bridge rewards mild.
    score = affinity * (
        0.55
        + 0.25 * reciprocal_rank
        + 0.20 * regret_pair
    )
    score *= 1.0 + 0.18 * mutual_top_k
    score *= 1.0 + 0.22 * mutual_nearest
    score *= 1.0 + 0.10 * bridge

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    # Ensure every node retains at least one sampling-supported edge.
    for i in range(n):
        if np.any(score[i] > 0.0):
            continue
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            j = candidates[np.argmin(distances[i, candidates])]
            score[i, j] = 1.0
            score[j, i] = 1.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
