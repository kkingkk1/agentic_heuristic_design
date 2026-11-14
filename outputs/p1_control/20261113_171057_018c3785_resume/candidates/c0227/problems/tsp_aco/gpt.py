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
        global_scale = float(np.median(positive))
        low_scale = float(np.percentile(positive, 25))
        high_scale = float(np.percentile(positive, 75))
    else:
        global_scale = low_scale = high_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)
    low_scale = max(low_scale, eps)
    high_scale = max(high_scale, global_scale)

    finite_rows = np.where(valid, work, np.inf)
    sorted_distances = np.sort(finite_rows, axis=1)
    nearest = sorted_distances[:, 0]

    if n > 2:
        second = sorted_distances[:, 1]
        third = sorted_distances[:, 2]
    else:
        second = nearest.copy()
        third = second.copy()

    row_scale = 0.55 * nearest + 0.30 * second + 0.15 * global_scale
    row_scale = np.maximum(row_scale, 0.08 * global_scale)

    # Robust soft distance preference.
    normalized_distance = np.where(
        valid,
        work / (0.90 * global_scale + 0.35 * row_scale[:, None]),
        0.0,
    )
    distance_term = 1.0 / np.sqrt(1.0 + normalized_distance)
    distance_term[~valid] = 0.0

    # Endpoint-relative locality with adaptive vertex scales.
    relative_cost = 0.5 * (
        work / row_scale[:, None] + work / row_scale[None, :]
    )
    locality_term = np.exp(-0.48 * np.minimum(relative_cost, 50.0))
    locality_term[~valid] = 0.0

    # Candidate ranks from both endpoints.
    order = np.argsort(finite_rows, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.25, 0.72 * np.sqrt(float(max(n, 2))))
    rank_term = np.exp(
        -0.72 * (ranks / rank_scale + ranks.T / rank_scale)
    )
    rank_term[~valid] = 0.0

    # Scarcity and regret: protect vertices whose best incident edges
    # are much more valuable than their alternatives.
    regret = np.clip(
        (second - nearest) / np.maximum(second, global_scale * 1e-12),
        0.0,
        1.0,
    )
    scarcity = np.clip(
        global_scale / np.maximum(row_scale, global_scale * 1e-12),
        0.25,
        2.5,
    )

    regret_term = 1.0 + 0.32 * np.sqrt(
        np.maximum(regret[:, None], 0.0)
        * np.maximum(regret[None, :], 0.0)
    )
    scarcity_term = 1.0 + 0.10 * np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    # Mutual-nearest-neighbor and reciprocal candidate bonuses.
    nearest_neighbor = np.argmin(finite_rows, axis=1)
    mutual = nearest_neighbor[nearest_neighbor] == np.arange(n)

    mutual_bonus = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    cols = nearest_neighbor
    mutual_bonus[rows, cols] = mutual.astype(float)
    mutual_bonus[cols, rows] = mutual.astype(float)

    reciprocal_rank = (
        (ranks <= 1.0) & (ranks.T <= 1.0) & valid
    ).astype(float)

    structural_term = (
        1.0
        + 0.24 * mutual_bonus
        + 0.10 * reciprocal_rank
    )

    # Local triangle consistency: edges supported by compatible third vertices
    # receive a modest bonus, while isolated long edges are not eliminated.
    k = min(max(2, int(np.sqrt(n))), max(1, n - 1))
    local_mask = ranks < float(k)

    local_scale = np.maximum(row_scale, global_scale * 1e-6)
    affinity = np.exp(
        -np.minimum(
            work / (0.75 * local_scale[:, None] + 0.25 * global_scale),
            40.0,
        )
    )
    affinity[~valid] = 0.0

    support = affinity @ affinity.T
    support_diag = np.maximum(np.diag(support), eps)
    support_normalized = support / np.sqrt(
        support_diag[:, None] * support_diag[None, :]
    )
    support_term = 1.0 + 0.12 * np.clip(support_normalized, 0.0, 1.0)
    support_term[~valid] = 1.0

    # Prim-style MST bridge signal.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
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
            v = parent[u]
            mst_bonus[u, v] = 1.0
            mst_bonus[v, u] = 1.0

        better = (~in_tree) & valid[u] & (work[u] < best)
        best[better] = work[u, better]
        parent[better] = u

    bridge_term = 1.0 + 0.10 * mst_bonus

    # Combine in log space to avoid excessive suppression from multiplication.
    components = [
        np.maximum(distance_term, eps),
        np.maximum(locality_term, eps),
        np.maximum(rank_term, eps),
        np.maximum(regret_term, eps),
        np.maximum(scarcity_term, eps),
        np.maximum(structural_term, eps),
        np.maximum(support_term, eps),
        np.maximum(bridge_term, eps),
    ]
    weights = np.array([0.80, 1.15, 1.05, 0.42, 0.20, 0.55, 0.28, 0.18])

    log_score = np.zeros((n, n), dtype=float)
    for weight, component in zip(weights, components):
        log_score += weight * np.log(component)

    score = np.exp(np.clip(log_score, -45.0, 45.0))
    score[~valid] = 0.0

    # Preserve a small but meaningful chance for feasible nonlocal edges.
    positive_scores = score[valid & (score > 0.0)]
    if positive_scores.size:
        floor = max(
            float(np.percentile(positive_scores, 10)) * 0.06,
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
