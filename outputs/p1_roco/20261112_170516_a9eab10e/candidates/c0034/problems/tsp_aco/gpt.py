import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-prior matrix based on leave-one-out MST consensus."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(raw)
    valid = finite & (raw >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    # TSP distances are treated as undirected. Invalid asymmetric entries
    # remain invalid unless both directions provide a usable value.
    undirected_valid = valid & valid.T
    d = 0.5 * (raw + raw.T)
    d = np.where(undirected_valid, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    cost_scale = float(np.median(positive)) if positive.size else 1.0
    cost_scale = max(cost_scale, np.finfo(float).eps)

    support = np.zeros((n, n), dtype=float)
    cost_accumulator = np.zeros((n, n), dtype=float)

    def add_edge(a: int, b: int) -> None:
        support[a, b] += 1.0
        support[b, a] += 1.0
        edge_cost = d[a, b]
        if np.isfinite(edge_cost):
            cost_accumulator[a, b] += edge_cost
            cost_accumulator[b, a] += edge_cost

    # Build an MST on every graph obtained by removing one possible root.
    # Prim's algorithm is used without introducing any external dependency.
    for excluded in range(n):
        active = np.ones(n, dtype=bool)
        active[excluded] = False

        if np.count_nonzero(active) < 2:
            continue

        in_tree = np.zeros(n, dtype=bool)
        best = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)

        start = int(np.flatnonzero(active)[0])
        best[start] = 0.0

        for _ in range(n - 1):
            candidates = np.where(active & ~in_tree, best, np.inf)
            current = int(np.argmin(candidates))

            if not np.isfinite(candidates[current]):
                break

            in_tree[current] = True
            predecessor = parent[current]

            if predecessor >= 0:
                add_edge(predecessor, current)

            outgoing = d[current]
            improve = active & ~in_tree & (outgoing < best)
            best[improve] = outgoing[improve]
            parent[improve] = current

    scenarios = float(max(n, 1))
    frequency = support / scenarios

    mean_cost = np.divide(
        cost_accumulator,
        support,
        out=np.full_like(cost_accumulator, np.inf),
        where=support > 0,
    )

    finite_mean = mean_cost[np.isfinite(mean_cost)]
    reference = (
        float(np.median(finite_mean))
        if finite_mean.size
        else cost_scale
    )
    reference = max(reference, np.finfo(float).eps)

    # Stable edges receive high support; among equally stable edges, lower
    # average MST cost receives a larger structural confidence.
    cost_confidence = np.divide(
        reference,
        reference + np.maximum(mean_cost, 0.0),
        out=np.zeros_like(mean_cost),
        where=np.isfinite(mean_cost),
    )

    score = frequency * (0.35 + 0.65 * cost_confidence)

    # A small tie-preserving correction favors edges appearing in more
    # distinct leave-one-out scenarios rather than repeated duplicate counts.
    scenario_fraction = np.minimum(support / scenarios, 1.0)
    score *= 0.5 + 0.5 * scenario_fraction

    score[~undirected_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
