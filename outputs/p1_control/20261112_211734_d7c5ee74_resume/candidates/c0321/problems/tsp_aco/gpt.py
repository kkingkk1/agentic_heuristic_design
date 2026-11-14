import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally scaled, connectivity-aware TSP edge prior."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    d = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    global_median = max(float(np.median(positive)), eps)
    global_q75 = max(float(np.percentile(positive, 75.0)), global_median)
    global_q90 = max(float(np.percentile(positive, 90.0)), global_q75)

    # Prim minimum spanning tree.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    mst = np.zeros((n, n), dtype=bool)

    for _ in range(n):
        masked_best = np.where(in_tree, np.inf, best)
        node = int(np.argmin(masked_best))

        if not np.isfinite(masked_best[node]):
            remaining = np.flatnonzero(~in_tree)
            if remaining.size == 0:
                break
            node = int(remaining[0])

        in_tree[node] = True

        if parent[node] >= 0:
            left = parent[node]
            mst[left, node] = True
            mst[node, left] = True

        candidates = ~in_tree
        improved = candidates & (work[node] < best)
        best[improved] = work[node, improved]
        parent[improved] = node

    mst_degree = np.sum(mst, axis=1).astype(float)

    # Global smooth distance preference.
    scaled_distance = d / global_q75
    distance_term = np.exp(-np.clip(scaled_distance ** 0.82, 0.0, 40.0))

    # Prefer edges that help low-degree tree vertices complete their degree budget.
    degree_need = np.maximum(0.0, 2.0 - mst_degree)
    degree_balance = 1.0 + 0.22 * (
        degree_need[:, None] + degree_need[None, :]
    ) / 2.0

    # Smoothly reward edges in the connectivity backbone without making it exclusive.
    tree_term = np.where(mst, 1.42, 0.78)

    # Penalize very long global outliers using a bounded tail factor.
    tail_ratio = np.maximum(d - global_q90, 0.0) / global_q90
    tail_term = 1.0 / (1.0 + 0.65 * tail_ratio)

    # Endpoint centrality is global rather than local-neighborhood based.
    row_mass = np.sum(np.where(np.isfinite(work), work, 0.0), axis=1)
    centrality = 1.0 / np.maximum(row_mass / max(float(n - 1), 1.0), eps)
    centrality /= max(float(np.max(centrality)), eps)
    centrality_term = 0.90 + 0.10 * np.sqrt(
        centrality[:, None] * centrality[None, :]
    )

    # Mild preference for edges below the global median.
    quantile_term = 0.88 + 0.12 * np.exp(
        -np.clip(d / global_median, 0.0, 30.0)
    )

    score = (
        distance_term
        * tree_term
        * degree_balance
        * tail_term
        * centrality_term
        * quantile_term
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
