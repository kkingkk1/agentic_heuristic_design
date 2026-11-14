import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic edge-priority indicators for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distances must be nonnegative")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    work = distances.copy()
    np.fill_diagonal(work, np.inf)
    valid = np.isfinite(work)

    positive = work[valid & (work > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0
    scale = max(scale, np.finfo(float).eps)

    finite_rows = np.where(valid, work, np.inf)
    nearest = np.min(finite_rows, axis=1)

    if n >= 3:
        second = np.partition(finite_rows, 1, axis=1)[:, 1]
    else:
        second = nearest.copy()

    local_scale = 0.5 * (nearest + second)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, scale * 1e-6)

    safe_work = np.where(valid, work, 0.0)

    # Robust inverse-distance preference.
    inverse_distance = 1.0 / (1.0 + safe_work / scale)
    inverse_distance[~valid] = 0.0

    # Relative endpoint cost, damped to avoid over-penalizing outliers.
    relative_cost = 0.5 * (
        safe_work / local_scale[:, None]
        + safe_work / local_scale[None, :]
    )
    relative_cost = np.minimum(relative_cost, 50.0)
    relative_cost_term = np.exp(-0.42 * relative_cost)
    relative_cost_term[~valid] = 0.0

    # Candidate rank at both endpoints.
    order = np.argsort(finite_rows, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values

    rank_scale = max(1.5, np.sqrt(float(n)))
    rank_term = np.exp(
        -0.5 * (ranks / rank_scale + ranks.T / rank_scale)
    )
    rank_term[~valid] = 0.0

    # Vertex regret: preserve edges needed by vertices with few good alternatives.
    gap = np.maximum(second - nearest, 0.0)
    regret = gap / np.maximum(second, scale * 1e-12)
    regret = np.clip(regret, 0.0, 1.0)

    endpoint_regret = np.sqrt(
        regret[:, None] * regret[None, :]
    )
    regret_term = 1.0 + 0.85 * endpoint_regret
    regret_term[~valid] = 0.0

    # Endpoint scarcity: an edge is more valuable if it is among the
    # few alternatives available to either endpoint.
    scarcity = 1.0 / (1.0 + np.minimum(ranks, ranks.T))
    scarcity_term = 1.0 + 0.30 * scarcity
    scarcity_term[~valid] = 0.0

    # Mutual nearest-neighbor structure.
    nearest_neighbor = np.argmin(finite_rows, axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)

    for u in range(n):
        v = int(nearest_neighbor[u])
        if v != u and nearest_neighbor[v] == u:
            mutual_bonus[u, v] = 1.0
            mutual_bonus[v, u] = 1.0

    mutual_term = 1.0 + 0.28 * mutual_bonus
    mutual_term[~valid] = 0.0

    # Prim-style minimum spanning tree bridge prior.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0
    mst_bonus = np.zeros((n, n), dtype=float)

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        u = int(np.argmin(candidates))

        if not np.isfinite(candidates[u]):
            break

        in_tree[u] = True

        if parent[u] >= 0:
            v = int(parent[u])
            mst_bonus[u, v] = 1.0
            mst_bonus[v, u] = 1.0

        better = (~in_tree) & valid[u] & (work[u] < best)
        best[better] = work[u, better]
        parent[better] = u

    mst_term = 1.0 + 0.14 * mst_bonus
    mst_term[~valid] = 0.0

    # Geometric blending keeps all signals active without allowing one
    # multiplicative factor to eliminate otherwise useful edges.
    safe_terms = [
        np.maximum(inverse_distance, 1e-12),
        np.maximum(relative_cost_term, 1e-12),
        np.maximum(rank_term, 1e-12),
        np.maximum(regret_term, 1e-12),
        np.maximum(scarcity_term, 1e-12),
        np.maximum(mutual_term, 1e-12),
        np.maximum(mst_term, 1e-12),
    ]
    weights = np.array([0.27, 0.19, 0.18, 0.13, 0.09, 0.08, 0.06])

    log_score = np.zeros((n, n), dtype=float)
    for weight, term in zip(weights, safe_terms):
        log_score += weight * np.log(term)

    score = np.exp(np.clip(log_score, -50.0, 50.0))
    score[~valid] = 0.0

    # Preserve feasible alternatives with a small adaptive floor.
    positive_scores = score[valid & (score > 0.0)]
    if positive_scores.size:
        floor = max(
            float(np.percentile(positive_scores, 15)) * 0.08,
            1e-12,
        )
        score[valid] = np.maximum(score[valid], floor)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~valid] = 0.0

    off_diagonal = ~np.eye(n, dtype=bool)
    mean_score = float(np.mean(score[off_diagonal]))

    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score
