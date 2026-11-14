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
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    ordered = np.sort(work, axis=1)
    k = min(max(2, int(np.ceil(np.sqrt(n)))), n - 1)

    nearest = ordered[:, 0]
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(
        local_scale,
        max(global_scale * 1e-6, np.finfo(float).eps),
    )

    second = ordered[:, 1] if n > 2 else nearest
    regret = np.maximum(second - nearest, 0.0)
    regret = regret / np.maximum(second, global_scale * 1e-12)
    regret_weight = np.sqrt(np.outer(regret, regret))

    safe_work = np.where(valid, work, 0.0)

    # Robust global preference for short edges.
    global_term = -np.log1p(safe_work / global_scale)

    # Endpoint-relative preference, using local scales at both endpoints.
    relative = 0.5 * (
        safe_work / local_scale[:, None]
        + safe_work / local_scale[None, :]
    )
    local_term = -np.minimum(relative, 40.0)

    # Rank preference at both endpoints.
    order = np.argsort(work, axis=1)
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_scale = max(1.5, np.sqrt(float(n)))
    rank_term = -0.5 * (
        ranks / rank_scale + ranks.T / rank_scale
    )

    # Preserve edges incident to vertices with distinctive nearest choices.
    regret_term = 0.55 * regret_weight

    nearest_neighbor = np.argmin(work, axis=1)
    mutual = (
        nearest_neighbor[nearest_neighbor] == np.arange(n)
    )
    mutual_bonus = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    cols = nearest_neighbor
    mutual_bonus[rows, cols] = mutual.astype(float)
    mutual_bonus[cols, rows] = mutual.astype(float)

    # Prim-style MST bridge identification.
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

    log_score = (
        0.95 * global_term
        + 0.85 * local_term
        + 0.65 * rank_term
        + regret_term
        + np.log1p(0.30 * mutual_bonus)
        + np.log1p(0.12 * mst_bonus)
    )

    log_score[~valid] = -np.inf
    finite_log = log_score[valid]

    if finite_log.size:
        reference = float(np.max(finite_log))
        score = np.exp(np.clip(log_score - reference, -40.0, 0.0))
    else:
        score = np.zeros((n, n), dtype=float)

    score[~valid] = 0.0

    positive_scores = score[valid & (score > 0.0)]
    if positive_scores.size:
        floor = max(
            float(np.percentile(positive_scores, 10)) * 0.05,
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
