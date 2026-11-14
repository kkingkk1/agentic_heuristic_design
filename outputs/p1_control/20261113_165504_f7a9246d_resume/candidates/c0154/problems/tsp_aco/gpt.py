import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return ensemble edge priors from perturbed global cheapest-link tours."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0.0):
        raise ValueError("distance_matrix must be nonnegative")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    if n == 2:
        if d[0, 1] >= 0.0:
            score[0, 1] = score[1, 0] = 1.0
        return score

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    rows, cols = np.triu_indices(n, k=1)
    edge_costs = d[rows, cols]
    edge_count = len(rows)

    trials = max(16, min(48, 3 * n))
    frequencies = np.zeros((n, n), dtype=float)

    for trial in range(trials):
        phase = 0.754877666 * (trial + 1)
        edge_index = np.arange(edge_count, dtype=float)

        # Deterministic perturbation creates diverse but reproducible edge orders.
        jitter = (
            0.035 * scale
            * (
                np.sin(edge_index * 1.6180339887 + phase)
                + 0.5 * np.cos(edge_index * 2.4142135624 - phase)
            )
        )
        perturbed = edge_costs + jitter

        order = np.argsort(perturbed, kind="stable")
        degree = np.zeros(n, dtype=int)
        parent = np.arange(n, dtype=int)
        size = np.ones(n, dtype=int)
        chosen = []

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        for position in order:
            u = int(rows[position])
            v = int(cols[position])

            if degree[u] >= 2 or degree[v] >= 2:
                continue

            same_component = find(u) == find(v)

            # Before n-1 edges, reject premature subtour formation.
            if same_component and len(chosen) < n - 1:
                continue

            chosen.append((u, v))
            degree[u] += 1
            degree[v] += 1

            if not same_component:
                union(u, v)

            if len(chosen) == n:
                break

        # Complete any unresolved degree slots using the original costs.
        if len(chosen) < n:
            fallback_order = np.argsort(edge_costs, kind="stable")
            chosen_set = {
                (min(u, v), max(u, v))
                for u, v in chosen
            }

            for position in fallback_order:
                u = int(rows[position])
                v = int(cols[position])
                key = (u, v)

                if key in chosen_set:
                    continue
                if degree[u] >= 2 or degree[v] >= 2:
                    continue

                chosen.append((u, v))
                chosen_set.add(key)
                degree[u] += 1
                degree[v] += 1

                if len(chosen) == n:
                    break

        for u, v in chosen:
            frequencies[u, v] += 1.0
            frequencies[v, u] += 1.0

    frequencies /= float(trials)

    # Continuous global fallback prevents unselected edges from becoming impossible.
    kernel = np.exp(-np.minimum(d / scale, 40.0))
    kernel[~valid] = 0.0

    score = 0.88 * frequencies + 0.12 * kernel
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    mean_score = float(np.mean(score[valid]))
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score.astype(float, copy=False)
