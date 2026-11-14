import numpy as np


def _kruskal_edge_mask(weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n = weights.shape[0]
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    selected = np.zeros(edges.shape[0], dtype=bool)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False

        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a

        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1
        return True

    order = np.argsort(
        weights[edges[:, 0], edges[:, 1]],
        kind="stable",
    )

    components = n
    for edge_index in order:
        i, j = edges[edge_index]
        if union(int(i), int(j)):
            selected[edge_index] = True
            components -= 1
            if components <= 1:
                break

    return selected


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global connectivity priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & ~diagonal
    nonnegative = np.maximum(np.where(finite, raw, np.nan), 0.0)

    available = finite | finite.T
    if not np.any(available):
        return np.zeros((n, n), dtype=float)

    forward = np.where(finite, nonnegative, 0.0)
    backward = np.where(finite.T, nonnegative.T, 0.0)
    counts = finite.astype(float) + finite.T.astype(float)

    distances = np.divide(
        forward + backward,
        np.maximum(counts, 1.0),
    )
    distances = 0.5 * (distances + distances.T)

    positive = distances[available & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(distances[available])) if np.any(available) else scale
    penalty = largest + 32.0 * scale

    valid = available.copy()
    distances[~valid] = penalty
    np.fill_diagonal(distances, penalty)

    edge_pairs = np.array(
        [(i, j) for i in range(n) for j in range(i + 1, n) if valid[i, j]],
        dtype=int,
    )

    if edge_pairs.size == 0:
        return np.zeros((n, n), dtype=float)

    edge_count = edge_pairs.shape[0]
    tree_count = max(8, min(32, 2 * n))
    occurrence = np.zeros(edge_count, dtype=float)
    degree_total = np.zeros(n, dtype=float)

    edge_values = distances[edge_pairs[:, 0], edge_pairs[:, 1]]
    value_scale = max(scale, np.finfo(float).eps)

    for trial in range(tree_count):
        phase = (trial + 1.0) * 0.6180339887498949
        edge_index = np.arange(edge_count, dtype=float) + 1.0

        jitter = 1.0 + 0.025 * np.sin(
            edge_index * (phase + 0.37)
            + 0.5 * np.cos(edge_index * 0.73)
        )
        weights = distances.copy()
        perturbed_values = np.maximum(edge_values * jitter, 0.0)
        weights[
            edge_pairs[:, 0],
            edge_pairs[:, 1],
        ] = perturbed_values
        weights[
            edge_pairs[:, 1],
            edge_pairs[:, 0],
        ] = perturbed_values

        selected = _kruskal_edge_mask(weights, edge_pairs)
        occurrence += selected.astype(float)

        degrees = np.zeros(n, dtype=float)
        chosen_edges = edge_pairs[selected]
        if chosen_edges.size:
            np.add.at(degrees, chosen_edges[:, 0], 1.0)
            np.add.at(degrees, chosen_edges[:, 1], 1.0)
        degree_total += degrees

    frequency = occurrence / float(tree_count)
    mean_degree = degree_total / float(tree_count)

    degree_balance = np.exp(
        -0.45 * np.abs(mean_degree[edge_pairs[:, 0]] - 2.0)
        -0.45 * np.abs(mean_degree[edge_pairs[:, 1]] - 2.0)
    )

    distance_affinity = np.exp(
        -edge_values / (0.80 * value_scale)
    )

    stability = np.sqrt(
        np.maximum(frequency, 0.0)
        * np.maximum(frequency, 0.0)
    )

    score_values = (
        0.62 * stability
        + 0.23 * distance_affinity
        + 0.15 * stability * degree_balance
    )

    score = np.zeros((n, n), dtype=float)
    score[
        edge_pairs[:, 0],
        edge_pairs[:, 1],
    ] = score_values
    score[
        edge_pairs[:, 1],
        edge_pairs[:, 0],
    ] = score_values

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
