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

    safe = np.where(valid, work, 0.0)

    # Robust base preference for short edges.
    base = 1.0 / (1.0 + safe / scale)
    base[~valid] = 0.0

    # Rank preference: locally high-ranked edges receive larger values.
    order = np.argsort(work, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_weight = np.exp(-ranks / max(2.0, np.sqrt(n)))
    rank_weight[~valid] = 0.0

    # Regret emphasizes edges that are much better than alternatives.
    sorted_distances = np.sort(np.where(valid, work, np.inf), axis=1)
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else nearest
    gap = np.maximum(second - nearest, 0.0)
    regret_i = np.where(
        np.isfinite(nearest),
        gap / (second + scale * 1e-12),
        0.0,
    )

    regret = np.sqrt(
        np.maximum(regret_i[:, None], 0.0)
        * np.maximum(regret_i[None, :], 0.0)
    )
    regret_term = 1.0 + 0.8 * regret

    nearest_neighbor = np.argmin(work, axis=1)
    mutual = (
        nearest_neighbor[nearest_neighbor] == np.arange(n)
    ).astype(float)
    mutual_bonus = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    cols = nearest_neighbor
    mutual_bonus[rows, cols] = mutual
    mutual_bonus[cols, rows] = mutual

    # Prim-style MST membership supplies a mild global connectivity prior.
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
    score *= np.sqrt(rank_weight * rank_weight.T)
    score *= regret_term
    score *= 1.0 + 0.35 * mutual_bonus
    score *= 1.0 + 0.15 * mst_bonus

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~valid] = 0.0

    off_diagonal = ~np.eye(n, dtype=bool)
    mean_score = float(np.mean(score[off_diagonal]))
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score
