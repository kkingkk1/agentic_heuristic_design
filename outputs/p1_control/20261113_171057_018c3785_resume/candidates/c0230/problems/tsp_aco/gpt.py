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

    work = 0.5 * (distances + distances.T)
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work)
    finite_values = work[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = max(float(np.mean(finite_values)), 1.0)
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    row_values = np.where(valid, work, np.inf)
    nearest = np.min(row_values, axis=1)

    if n > 2:
        second = np.partition(row_values, 1, axis=1)[:, 1]
    else:
        second = nearest.copy()

    fallback = np.where(
        np.isfinite(nearest),
        nearest,
        global_scale,
    )
    endpoint_scale = np.where(
        np.isfinite(second),
        0.5 * (fallback + second),
        global_scale,
    )
    endpoint_scale = np.maximum(endpoint_scale, global_scale * 1e-6)

    safe_work = np.where(valid, work, 0.0)

    # Robust global distance affinity with a soft tail.
    distance_affinity = 1.0 / (
        1.0 + safe_work / (1.25 * global_scale)
    )
    distance_affinity[~valid] = 0.0

    # Endpoint-relative locality rewards edges that are good for both vertices.
    relative_cost = 0.5 * (
        safe_work / endpoint_scale[:, None]
        + safe_work / endpoint_scale[None, :]
    )
    locality = np.exp(-0.72 * np.minimum(relative_cost, 40.0))
    locality[~valid] = 0.0

    # Rank quality is computed independently at both endpoints.
    order = np.argsort(row_values, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.25, 0.75 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.68 * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
    )
    rank_quality[~valid] = 0.0

    # Regret identifies vertices for which losing the best edge is costly.
    denominator = np.maximum(second, np.maximum(nearest, global_scale * 1e-12))
    regret_vertex = np.clip(
        (second - nearest) / denominator,
        0.0,
        1.0,
    )
    regret = np.sqrt(
        regret_vertex[:, None] * regret_vertex[None, :]
    )
    regret[~valid] = 0.0

    # Mutual nearest-neighbor edges receive a moderate structural bonus.
    nearest_neighbor = np.argmin(row_values, axis=1)
    mutual = nearest_neighbor[nearest_neighbor] == np.arange(n)

    mutual_bonus = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    cols = nearest_neighbor
    mutual_bonus[rows, cols] = mutual.astype(float)
    mutual_bonus[cols, rows] = mutual.astype(float)
    mutual_bonus[~valid] = 0.0

    # Prim-style MST edges preserve globally useful bridge connections.
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

    mst_bonus[~valid] = 0.0

    # Use additive evidence to avoid excessive suppression of distinctive edges.
    evidence = (
        0.34 * distance_affinity
        + 0.27 * locality
        + 0.25 * rank_quality
        + 0.08 * regret
    )
    evidence *= 1.0 + 0.30 * mutual_bonus
    evidence *= 1.0 + 0.12 * mst_bonus
    evidence[~valid] = 0.0

    positive = evidence[valid & (evidence > 0.0)]

    if positive.size:
        floor = max(
            float(np.percentile(positive, 10.0)) * 0.06,
            np.finfo(float).eps,
        )
        evidence[valid] = np.maximum(evidence[valid], floor)

    score = 0.5 * (evidence + evidence.T)
    np.fill_diagonal(score, 0.0)
    score[~valid] = 0.0

    off_diagonal = ~np.eye(n, dtype=bool)
    mean_score = float(np.mean(score[off_diagonal]))

    if np.isfinite(mean_score) and mean_score > 0.0:
        score /= mean_score

    return score
