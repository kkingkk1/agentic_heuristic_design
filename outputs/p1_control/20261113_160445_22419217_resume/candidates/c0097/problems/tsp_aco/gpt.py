import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a topology-aware symmetric edge-priority matrix for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must be nonnegative")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    # TSP edge priors are made explicitly undirected.
    d = 0.5 * (distances + distances.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Build a minimum spanning tree with Prim's algorithm.
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    mst = np.zeros((n, n), dtype=bool)

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        u = int(np.argmin(candidates))

        if not np.isfinite(candidates[u]):
            break

        in_tree[u] = True
        v = parent[u]
        if v >= 0:
            mst[u, v] = True
            mst[v, u] = True

        improved = (~in_tree) & (d[u] < best)
        best[improved] = d[u, improved]
        parent[improved] = u

    mst_degree = mst.sum(axis=1).astype(float)

    # Minimax path distance on the MST:
    # bottleneck[i, j] is the largest tree edge on the unique tree path.
    bottleneck = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(bottleneck, 0.0)
    bottleneck[mst] = d[mst]

    for k in range(n):
        bottleneck = np.minimum(
            bottleneck,
            np.maximum(
                bottleneck[:, k, None],
                bottleneck[None, k, :],
            ),
        )

    finite_bottleneck = np.isfinite(bottleneck)
    detour_gap = np.maximum(d - bottleneck, 0.0)
    detour_gap[~finite_bottleneck] = 0.0

    # Edges close to the MST bottleneck form efficient alternative links.
    bottleneck_term = np.exp(
        -np.clip(detour_gap / (0.75 * scale), 0.0, 60.0)
    )
    bottleneck_term[~np.isfinite(bottleneck_term)] = 0.0

    # Give tree edges a structural anchor without making them mandatory.
    tree_term = 1.0 + 0.35 * mst.astype(float)

    # Count short triangular closures. A third node supports (i, j) when
    # both incident edges are no longer than the candidate edge.
    finite_d = np.where(np.isfinite(d), d, 0.0)
    closure = np.zeros((n, n), dtype=float)

    for k in range(n):
        left = d[:, k][:, None]
        right = d[k, :][None, :]
        support = (left <= d) & (right <= d)
        excess = np.maximum(left + right - d, 0.0)
        contribution = np.exp(
            -np.clip(excess / scale, 0.0, 60.0)
        )
        closure += support * contribution

    np.fill_diagonal(closure, 0.0)
    closure_scale = np.max(closure)
    if closure_scale > 0.0 and np.isfinite(closure_scale):
        closure_term = closure / closure_scale
    else:
        closure_term = np.zeros_like(closure)

    # Prefer edges that do not consume both endpoints with highly branched
    # tree structure, which leaves more degree capacity for a Hamiltonian tour.
    degree_pressure = (mst_degree[:, None] + mst_degree[None, :]) / max(2.0, n)
    degree_term = np.exp(-0.55 * np.clip(degree_pressure, 0.0, 20.0))

    # A mild scale-free tie breaker based on the global distance distribution.
    distance_term = 1.0 / (
        1.0 + np.clip(finite_d / scale, 0.0, 1e12)
    )

    score = (
        0.48 * bottleneck_term
        + 0.24 * closure_term
        + 0.16 * distance_term
        + 0.12 * degree_term
    )
    score *= tree_term

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
