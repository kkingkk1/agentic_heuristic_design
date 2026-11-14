import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    fallback = float(np.max(finite_positive)) if finite_positive.size else 1.0

    clean = np.array(d, dtype=float, copy=True)
    invalid = ~np.isfinite(clean) | (clean < 0)
    clean[invalid] = fallback

    # Use a symmetric metric for undirected TSP structure.
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, np.inf)

    positive = clean[np.isfinite(clean) & (clean > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Smooth short-edge preference, less sensitive to extreme distances.
    smooth_inverse = 1.0 / (1.0 + clean / scale)
    smooth_inverse[~np.isfinite(smooth_inverse)] = 0.0

    # Rank-based neighborhood affinity.
    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values

    temperature = max(1.0, 0.18 * (n - 1))
    rank_affinity = np.exp(-ranks / temperature)
    rank_affinity[~np.isfinite(rank_affinity)] = 0.0
    np.fill_diagonal(rank_affinity, 0.0)

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Prim MST backbone.
    mst = np.zeros((n, n), dtype=float)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    for _ in range(n):
        candidate_cost = np.where(in_tree, np.inf, best)
        u = int(np.argmin(candidate_cost))
        if in_tree[u] or not np.isfinite(candidate_cost[u]):
            break

        in_tree[u] = True
        p = parent[u]
        if p >= 0:
            mst[u, p] = 1.0
            mst[p, u] = 1.0

        unvisited = ~in_tree
        improved = unvisited & (clean[u] < best)
        best[improved] = clean[u, improved]
        parent[improved] = u

    score = (
        0.46 * smooth_inverse
        + 0.30 * rank_affinity
        + 0.14 * mutual
        + 0.10 * mst
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
