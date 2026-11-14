import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric TSP edge priors from perturbed spanning-forest persistence."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    symmetric = 0.5 * (raw + raw.T)
    distances[valid] = symmetric[valid]
    np.fill_diagonal(distances, np.inf)

    edge_i, edge_j = np.triu_indices(n, k=1)
    edge_mask = valid[edge_i, edge_j]
    edge_i = edge_i[edge_mask]
    edge_j = edge_j[edge_mask]
    edge_values = distances[edge_i, edge_j]

    if edge_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = edge_values[edge_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    order = np.argsort(edge_values, kind="stable")
    sorted_values = edge_values[order]

    ranks = np.empty(edge_values.size, dtype=float)
    ranks[order] = np.arange(edge_values.size, dtype=float)
    quantile = ranks / max(float(edge_values.size - 1), 1.0)

    local_kernel = np.exp(-2.4 * quantile)
    local_kernel *= np.exp(-0.35 * np.minimum(edge_values / scale, 40.0))

    repetitions = int(np.clip(8 + 2 * np.sqrt(n), 10, 28))
    persistence = np.zeros(edge_values.size, dtype=float)
    degree_presence = np.zeros(n, dtype=float)

    def forest_edges(weights):
        parent = np.arange(n, dtype=int)
        size = np.ones(n, dtype=int)
        selected = []

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return False
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            return True

        for index in np.argsort(weights, kind="stable"):
            a = int(edge_i[index])
            b = int(edge_j[index])
            if union(a, b):
                selected.append(index)

        return selected

    base = np.maximum(edge_values / scale, 0.0)

    for repetition in range(repetitions):
        phase = float(repetition + 1)
        perturbation = (
            0.035 * np.sin(
                (edge_i + 1.0) * (12.9898 + phase * 0.731)
                + (edge_j + 1.0) * (78.233 - phase * 0.217)
            )
            + 0.020 * np.cos(
                (edge_i + edge_j + 2.0) * (4.123 + phase * 0.419)
            )
        )

        weights = base * (1.0 + perturbation)
        weights += 1e-10 * (
            (edge_i + 1.0) / max(float(n), 1.0)
            + (edge_j + 1.0) / max(float(n * n), 1.0)
        )

        selected = forest_edges(weights)
        for index in selected:
            persistence[index] += 1.0
            degree_presence[edge_i[index]] += 1.0
            degree_presence[edge_j[index]] += 1.0

    persistence /= float(repetitions)
    degree_presence /= float(repetitions)

    node_support = np.maximum(degree_presence, 1e-12)
    node_support /= max(float(np.mean(node_support[node_support > 0])), 1.0)

    coverage = np.sqrt(node_support[edge_i] * node_support[edge_j])
    coverage = np.clip(coverage, 0.35, 1.65)

    persistence_term = np.power(np.maximum(persistence, 0.0), 1.35)
    score_edges = (
        0.78 * persistence_term
        + 0.17 * persistence_term * local_kernel
        + 0.05 * local_kernel
    )
    score_edges *= 0.85 + 0.15 * coverage

    result = np.zeros((n, n), dtype=float)
    result[edge_i, edge_j] = score_edges
    result[edge_j, edge_i] = score_edges
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result[valid] = 1.0
        np.fill_diagonal(result, 0.0)
        return result

    result /= maximum
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
