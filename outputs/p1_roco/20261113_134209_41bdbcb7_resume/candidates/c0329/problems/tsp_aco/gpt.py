import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    forward = np.where(finite_nonnegative, distances, np.inf)
    backward = forward.T

    forward_ok = np.isfinite(forward)
    backward_ok = np.isfinite(backward)
    both_ok = forward_ok & backward_ok

    distance = np.full((n, n), np.inf, dtype=float)

    # Use a conservative symmetrization for asymmetric inputs.
    distance[both_ok] = 0.5 * (
        forward[both_ok] + backward[both_ok]
    )
    distance[forward_ok & ~backward_ok] = forward[
        forward_ok & ~backward_ok
    ]
    distance[backward_ok & ~forward_ok] = backward[
        backward_ok & ~forward_ok
    ]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance) & ~diagonal

    if not np.any(valid):
        return result

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    safe = np.where(valid, distance, np.inf)

    with np.errstate(over="ignore", invalid="ignore"):
        direct_affinity = np.exp(
            -np.clip(safe / scale, 0.0, 50.0)
        )

    direct_affinity[~valid] = 0.0

    # Soft accessibility measures how many nearby alternatives each node has.
    accessibility = np.sum(direct_affinity, axis=1)
    accessibility = np.maximum(accessibility, eps)

    node_sparsity = 1.0 / np.sqrt(accessibility)
    node_sparsity /= max(float(np.max(node_sparsity)), eps)

    # Find the cheapest valid two-hop replacement for every edge.
    hop_distance = safe[:, :, None] + safe[None, :, :]
    two_hop = np.min(hop_distance, axis=1)

    # Exclude the direct edge from the intermediate-node search.
    two_hop = np.where(np.isfinite(two_hop), two_hop, np.inf)
    detour_ratio = np.divide(
        two_hop,
        np.maximum(safe, eps),
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    # A large ratio means the edge is much shorter than any two-hop detour.
    irreplaceability = np.tanh(
        np.clip(detour_ratio - 1.0, 0.0, 8.0) / 2.0
    )
    irreplaceability[~valid] = 0.0

    # Prim-style minimum spanning forest support.
    forest_edge = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)

    for root in range(n):
        if visited[root]:
            continue

        visited[root] = True
        best = safe[root].copy()
        parent = np.full(n, root, dtype=int)
        best[root] = np.inf

        for _ in range(n - 1):
            candidates = np.where(~visited, best, np.inf)
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            previous = int(parent[node])
            if previous != node and np.isfinite(safe[node, previous]):
                forest_edge[node, previous] = True
                forest_edge[previous, node] = True

            visited[node] = True
            improve = (~visited) & (safe[node] < best)
            best[improve] = safe[node, improve]
            parent[improve] = node

    forest_support = forest_edge.astype(float)

    # Penalize ordinary edges in highly accessible regions, while retaining
    # a mild reward for edges incident to sparse endpoints.
    sparsity_support = np.sqrt(
        node_sparsity[:, None] * node_sparsity[None, :]
    )
    sparsity_support = 0.82 + 0.18 * sparsity_support

    # Edges that are both short and structurally hard to replace receive the
    # main score; the forest term is deliberately bounded.
    scores = (
        direct_affinity
        * (0.35 + 0.65 * irreplaceability)
        * (1.0 + 0.48 * forest_support)
        * sparsity_support
    )

    # If no finite two-hop detour exists, preserve direct-distance evidence.
    no_detour = valid & ~np.isfinite(two_hop)
    scores[no_detour] = (
        direct_affinity[no_detour]
        * (1.0 + 0.48 * forest_support[no_detour])
        * sparsity_support[no_detour]
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
