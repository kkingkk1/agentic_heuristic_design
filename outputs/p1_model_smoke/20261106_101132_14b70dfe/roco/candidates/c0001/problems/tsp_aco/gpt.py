import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term-aware edge desirability scores for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    finite_edges = work[np.isfinite(work)]
    positive_edges = finite_edges[finite_edges > 0]

    if positive_edges.size:
        q25, q50, q75 = np.percentile(positive_edges, [25.0, 50.0, 75.0])
        scale = max(float(q50), np.finfo(float).eps)
        spread = max(float(q75 - q25), np.finfo(float).eps)
    else:
        scale = 1.0
        spread = 1.0

    finite_mask = np.isfinite(work)

    # Multi-scale efficiency: short edges are preferred, but not exclusively.
    local_efficiency = 1.0 / (1.0 + work / scale)
    broad_efficiency = 1.0 / (1.0 + work / (scale + spread))
    efficiency = 0.60 * local_efficiency + 0.40 * broad_efficiency
    efficiency[~finite_mask] = 0.0

    # Global rank preference avoids overcommitting to only the shortest edges.
    ranks = np.zeros_like(work, dtype=float)
    for i in range(n):
        row = work[i]
        valid = np.isfinite(row)
        count = int(np.sum(valid))
        if count:
            order = np.argsort(row, kind="mergesort")
            rank_values = np.empty(n, dtype=float)
            rank_values[order] = np.arange(n, dtype=float)
            ranks[i, valid] = 1.0 - rank_values[valid] / max(count - 1, 1)
    rank_score = np.sqrt(np.maximum(ranks, 0.0))

    # Endpoint flexibility: retain edges useful to cities with few good options.
    ordered = np.sort(work, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else nearest.copy()
    third = ordered[:, 2] if n > 3 else second.copy()

    local_gap = np.maximum(second - nearest, 0.0)
    continuation_gap = np.maximum(third - second, 0.0)

    scarcity = nearest / (nearest + scale)
    scarcity_gap = local_gap / (local_gap + spread)
    continuation = continuation_gap / (continuation_gap + spread)

    endpoint_value = (
        0.45
        + 0.30 * scarcity
        + 0.15 * scarcity_gap
        + 0.10 * continuation
    )

    endpoint_score = (
        endpoint_value[:, None] + endpoint_value[None, :]
    ) / 2.0

    # Two-step potential: reward edges whose endpoints can continue efficiently.
    continuation_cost = nearest[:, None] + nearest[None, :]
    two_step_potential = continuation_cost / (
        continuation_cost + work + scale
    )
    two_step_potential[~finite_mask] = 0.0

    # Diversity bonus favors useful alternatives rather than duplicate nearest edges.
    nearest_threshold = nearest[:, None] + 0.25 * spread
    alternative = (work > nearest_threshold).astype(float)
    alternative[~finite_mask] = 0.0
    diversity_bonus = 0.75 + 0.25 * alternative

    scores = (
        0.34 * efficiency
        + 0.22 * rank_score
        + 0.18 * endpoint_score
        + 0.16 * two_step_potential
        + 0.10 * diversity_bonus
    )

    scores[~finite_mask] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    scores[~np.isfinite(scores)] = 0.0
    return scores
