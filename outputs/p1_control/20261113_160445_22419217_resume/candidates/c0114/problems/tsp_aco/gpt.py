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
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work)
    off_diagonal = ~np.eye(n, dtype=bool)

    positive = work[valid & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Robust short-edge preference with a bounded dynamic range.
    base = np.zeros_like(work)
    base[valid] = 1.0 / (1.0 + work[valid] / scale)

    # Exponential endpoint-relative rank preference.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = np.inf

    temperature = max(1.5, np.sqrt(float(n)))
    row_rank = np.exp(-ranks / temperature)
    row_rank[~valid] = 0.0

    rank_term = np.sqrt(row_rank * row_rank.T)

    # Regret identifies endpoints whose best alternatives are scarce.
    sorted_work = np.sort(np.where(valid, work, np.inf), axis=1)
    nearest = sorted_work[:, 0]

    if n >= 3:
        second = sorted_work[:, 1]
    else:
        second = nearest.copy()

    gap = np.maximum(second - nearest, 0.0)
    regret = np.zeros(n, dtype=float)
    finite_nearest = np.isfinite(nearest)
    regret[finite_nearest] = gap[finite_nearest] / (
        second[finite_nearest] + scale * 1e-12
    )

    regret = np.clip(regret, 0.0, 1.0)
    regret_term = 1.0 + 0.8 * np.sqrt(regret[:, None] * regret[None, :])

    # Mutual-nearest-neighbor structure.
    nearest_neighbor = np.argmin(work, axis=1)
    mutual = np.zeros_like(work)
    rows = np.arange(n)
    mutual_rows = nearest_neighbor[nearest_neighbor] == rows
    mutual[rows[mutual_rows], nearest_neighbor[mutual_rows]] = 1.0
    mutual[nearest_neighbor[mutual_rows], rows[mutual_rows]] = 1.0
    mutual_term = 1.0 + 0.35 * mutual

    # Prim-style MST membership preserves globally useful bridge edges.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0
    mst = np.zeros_like(work)

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        u = int(np.argmin(candidates))

        if not np.isfinite(candidates[u]):
            break

        in_tree[u] = True

        if parent[u] >= 0:
            v = parent[u]
            mst[u, v] = 1.0
            mst[v, u] = 1.0

        better = (~in_tree) & valid[u] & (work[u] < best)
        best[better] = work[u, better]
        parent[better] = u

    mst_term = 1.0 + 0.15 * mst

    # Normalize components by their valid off-diagonal means.
    components = [base, rank_term, regret_term, mutual_term, mst_term]
    normalized = []

    for component in components:
        component = np.where(valid, component, 0.0)
        mean_value = float(np.mean(component[off_diagonal]))

        if mean_value > 0.0 and np.isfinite(mean_value):
            component = component / mean_value

        normalized.append(component)

    base_n, rank_n, regret_n, mutual_n, mst_n = normalized

    # Multiplicative fusion rewards edges supported by several signals.
    score = (
        np.power(np.maximum(base_n, 0.0), 0.42)
        * np.power(np.maximum(rank_n, 0.0), 0.28)
        * np.power(np.maximum(regret_n, 0.0), 0.12)
        * np.power(np.maximum(mutual_n, 0.0), 0.10)
        * np.power(np.maximum(mst_n, 0.0), 0.08)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    mean_score = float(np.mean(score[off_diagonal]))
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score
