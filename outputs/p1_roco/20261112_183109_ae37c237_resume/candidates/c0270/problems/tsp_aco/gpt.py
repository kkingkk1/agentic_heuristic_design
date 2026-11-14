import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)
    valid = finite & finite.T

    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.max(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    largest = float(np.max(finite_values))
    fallback = max(largest, scale) * 4.0 + scale
    work = np.where(valid, d, fallback)
    np.fill_diagonal(work, fallback)

    # Smooth distance quality using a rational decay rather than exponentials.
    normalized_distance = work / scale
    distance_quality = 1.0 / (1.0 + normalized_distance) ** 2

    # Count cheaper incident edges. This measures endpoint competition.
    cheaper = valid[:, :, None] & (
        work[:, :, None] < work[:, None, :]
    )
    cheaper_count = np.sum(cheaper, axis=1).astype(float)

    endpoint_degree = np.sum(valid, axis=1).astype(float)
    degree_scale = np.maximum(endpoint_degree, 1.0)

    competition = (
        cheaper_count / degree_scale[:, None]
        + cheaper_count.T / degree_scale[None, :]
    )
    competition = np.clip(competition / 2.0, 0.0, 1.0)

    # Compute the best two-hop detour that does not use the direct edge.
    detour = np.full((n, n), np.inf, dtype=float)

    if n > 2:
        for h in range(n):
            left_valid = valid[:, h][:, None]
            right_valid = valid[h, :][None, :]
            through_h = np.maximum(
                work[:, h][:, None],
                work[h, :][None, :],
            )
            through_h[~(left_valid & right_valid)] = np.inf
            detour = np.minimum(detour, through_h)

    np.fill_diagonal(detour, np.inf)

    detour_ratio = np.divide(
        detour,
        np.maximum(work, eps),
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    # Criticality approaches one when every short detour is substantially worse.
    criticality = np.divide(
        detour_ratio - 1.0,
        detour_ratio + 1.0,
        out=np.ones((n, n), dtype=float),
        where=np.isfinite(detour_ratio),
    )
    criticality = np.clip(
        np.nan_to_num(
            criticality,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    # Approximate a minimum-spanning-tree skeleton with Kruskal's algorithm.
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    skeleton = np.zeros((n, n), dtype=bool)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(first: int, second: int) -> bool:
        root_first = find(first)
        root_second = find(second)

        if root_first == root_second:
            return False

        if rank[root_first] < rank[root_second]:
            parent[root_first] = root_second
        elif rank[root_first] > rank[root_second]:
            parent[root_second] = root_first
        else:
            parent[root_second] = root_first
            rank[root_first] += 1

        return True

    upper_i, upper_j = np.where(
        np.triu(valid, k=1)
    )

    if upper_i.size:
        edge_order = np.argsort(
            work[upper_i, upper_j],
            kind="stable",
        )

        for position in edge_order:
            i = int(upper_i[position])
            j = int(upper_j[position])
            if union(i, j):
                skeleton[i, j] = True
                skeleton[j, i] = True

    skeleton_bonus = np.where(skeleton, 1.55, 1.0)

    # Give structurally important edges a bounded boost while
    # reducing edges that have many cheaper local competitors.
    structural_bonus = (
        1.0
        + 0.72 * criticality
        + 0.18 * criticality * (1.0 - competition)
    )

    competition_penalty = 1.0 - 0.24 * competition

    scores = (
        distance_quality
        * structural_bonus
        * skeleton_bonus
        * competition_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
