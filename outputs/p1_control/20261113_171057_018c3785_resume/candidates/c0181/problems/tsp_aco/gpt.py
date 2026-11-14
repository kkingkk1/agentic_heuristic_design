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
    positive = work[valid & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    finite_rows = np.where(valid, work, np.inf)
    nearest = np.min(finite_rows, axis=1)
    second = np.partition(finite_rows, 1, axis=1)[:, 1] if n > 2 else nearest

    row_scale = np.where(
        np.isfinite(second),
        0.5 * (nearest + second),
        scale,
    )
    row_scale = np.maximum(row_scale, scale * 1e-6)

    # Softer robust short-edge preference.
    base = 1.0 / (1.0 + np.where(valid, work, 0.0) / (1.15 * scale))
    base[~valid] = 0.0

    # Endpoint-relative excess cost.
    relative_cost = 0.5 * (
        work / row_scale[:, None] + work / row_scale[None, :]
    )
    relative_cost = np.where(valid, relative_cost, 0.0)
    locality = np.exp(-0.60 * np.minimum(relative_cost, 40.0))
    locality[~valid] = 0.0

    # Rank-based candidate preference at both endpoints.
    order = np.argsort(finite_rows, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.35, 0.85 * np.sqrt(float(n)))
    rank_term = np.exp(
        -0.60 * (ranks / rank_scale + ranks.T / rank_scale)
    )
    rank_term[~valid] = 0.0

    # Regret emphasizes vertices with a strong best-edge preference.
    gap = np.maximum(second - nearest, 0.0)
    regret_vertex = gap / np.maximum(second, scale * 1e-12)
    regret_term = 1.0 + 0.85 * np.sqrt(
        np.maximum(regret_vertex[:, None], 0.0)
        * np.maximum(regret_vertex[None, :], 0.0)
    )

    # Mutual nearest-neighbor bonus.
    nearest_neighbor = np.argmin(finite_rows, axis=1)
    mutual = nearest_neighbor[nearest_neighbor] == np.arange(n)
    mutual_bonus = np.zeros((n, n), dtype=float)

    rows = np.arange(n)
    cols = nearest_neighbor
    mutual_bonus[rows, cols] = mutual.astype(float)
    mutual_bonus[cols, rows] = mutual.astype(float)

    # Prim-style MST bonus for globally important bridge edges.
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

    score = base
    score *= locality
    score *= rank_term
    score *= regret_term
    score *= 1.0 + 0.36 * mutual_bonus
    score *= 1.0 + 0.14 * mst_bonus

    # Preserve a small probability of selecting feasible nonpreferred edges.
    positive_scores = score[valid & (score > 0.0)]
    if positive_scores.size:
        floor = max(float(np.percentile(positive_scores, 12)) * 0.04, 1e-12)
        score[valid] = np.maximum(score[valid], floor)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~valid] = 0.0

    off_diagonal = ~np.eye(n, dtype=bool)
    mean_score = float(np.mean(score[off_diagonal]))

    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score
