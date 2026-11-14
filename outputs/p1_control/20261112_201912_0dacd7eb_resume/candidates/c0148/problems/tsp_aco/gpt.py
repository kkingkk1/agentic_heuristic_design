import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return TSP edge priors from an ensemble of minimum 1-trees."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    positive = raw[finite & (raw > 0.0)]
    fallback = float(np.max(positive)) if positive.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, raw, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    if n == 2:
        result = np.zeros((2, 2), dtype=float)
        result[0, 1] = result[1, 0] = 1.0
        return result

    edge_i, edge_j = np.triu_indices(n, 1)
    base_edges = np.column_stack((edge_i, edge_j))
    base_costs = d[edge_i, edge_j]

    support = np.zeros((n, n), dtype=float)
    degree_error = np.zeros(n, dtype=float)

    perturbations = max(4, min(12, n))
    scale = max(fallback, float(np.median(base_costs)), np.finfo(float).eps)

    def add_edge(i: int, j: int, degrees: np.ndarray, weight: float) -> None:
        if i == j:
            return
        support[i, j] += weight
        support[j, i] += weight
        degrees[i] += 1.0
        degrees[j] += 1.0

    for root in range(n):
        mask = (edge_i != root) & (edge_j != root)
        inner_edges = base_edges[mask]
        inner_costs = base_costs[mask]

        for p in range(perturbations):
            if inner_edges.shape[0] == 0:
                continue

            a = inner_edges[:, 0].astype(float) + 1.0
            b = inner_edges[:, 1].astype(float) + 1.0

            tie_signal = (
                np.sin((a + 0.37 * root) * (p + 1.0))
                + np.cos((b + 0.61 * root) * (p + 2.0))
                + np.sin((a + b) * (p + 3.0))
            )

            perturbed = inner_costs + (
                1e-10 * scale * tie_signal
            )

            order = np.lexsort(
                (
                    inner_edges[:, 1],
                    inner_edges[:, 0],
                    perturbed,
                )
            )

            parent = np.arange(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            degrees = np.zeros(n, dtype=float)
            chosen = []

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> bool:
                rx = find(x)
                ry = find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True

            for idx in order:
                u = int(inner_edges[idx, 0])
                v = int(inner_edges[idx, 1])
                if union(u, v):
                    chosen.append((u, v))
                    if len(chosen) == n - 2:
                        break

            for u, v in chosen:
                add_edge(u, v, degrees, 1.0)

            root_candidates = np.arange(n)
            root_candidates = root_candidates[root_candidates != root]
            root_costs = d[root, root_candidates]
            root_order = np.lexsort((root_candidates, root_costs))

            for idx in root_order[:2]:
                v = int(root_candidates[idx])
                add_edge(root, v, degrees, 1.0)

            degree_error += np.abs(degrees - 2.0)

    total_trials = float(n * perturbations)
    if total_trials > 0.0:
        support /= total_trials
        degree_error /= total_trials

    endpoint_penalty = 1.0 / (
        1.0
        + 0.5 * (degree_error[:, None] + degree_error[None, :])
    )

    score = support * endpoint_penalty
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        maximum = float(np.max(values))
        if np.isfinite(maximum) and maximum > 0.0:
            score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
