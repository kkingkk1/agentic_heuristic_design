import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, scarcity-aware edge desirability scores for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must be non-negative")

    scores = np.zeros_like(distances, dtype=float)
    if n == 1:
        return scores

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    finite_edges = work[np.isfinite(work)]
    positive_edges = finite_edges[finite_edges > 0]
    scale = (
        float(np.median(positive_edges))
        if positive_edges.size
        else 1.0
    )
    scale = max(scale, np.finfo(float).eps)

    # Robust global distance normalization.
    normalized = work / scale
    shortness = 1.0 / (1.0 + normalized)

    # Per-node order statistics and rank-based local support.
    ordered = np.sort(work, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else nearest.copy()
    third = ordered[:, 2] if n > 3 else second.copy()

    regret = np.maximum(second - nearest, 0.0)
    higher_regret = np.maximum(third - second, 0.0)

    regret_scale = np.median(regret[regret > 0]) if np.any(regret > 0) else scale
    regret_scale = max(float(regret_scale), np.finfo(float).eps)

    node_scarcity = nearest / (nearest + scale)
    node_regret = regret / (regret + regret_scale)
    node_future = higher_regret / (higher_regret + regret_scale)

    # Rank preference decreases smoothly, retaining alternatives for exploration.
    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    rank_decay = np.exp(-ranks / max(1.0, 0.35 * n))
    rank_support = 0.5 * (rank_decay + rank_decay.T)

    # Compare an edge against both endpoints' best alternatives.
    endpoint_best = nearest[:, None] + nearest[None, :]
    endpoint_regret = np.minimum(
        np.maximum(second[:, None] - work, 0.0),
        np.maximum(second[None, :] - work, 0.0),
    )
    endpoint_regret = endpoint_regret / (endpoint_regret + regret_scale)

    # Global bridge potential: favor edges whose length is moderate globally
    # but whose endpoints are locally isolated, preserving useful long links.
    finite_mask = np.isfinite(work)
    global_values = work[finite_mask]
    global_center = np.median(global_values) if global_values.size else scale
    global_spread = (
        np.median(np.abs(global_values - global_center))
        if global_values.size
        else scale
    )
    global_spread = max(float(global_spread), scale * 0.1, np.finfo(float).eps)

    bridge_distance = np.exp(
        -np.abs(work - global_center) / global_spread
    )
    isolation_pair = 0.5 * (
        node_scarcity[:, None] + node_scarcity[None, :]
    )
    future_pair = 0.5 * (
        node_future[:, None] + node_future[None, :]
    )
    bridge_potential = bridge_distance * (0.5 + isolation_pair) * (
        0.75 + 0.25 * future_pair
    )

    # Controlled exploratory bonus for edges outside the nearest-neighbor set.
    # It is bounded, so exploration cannot overwhelm geometric usefulness.
    exploratory = (
        np.sqrt(np.maximum(normalized, 0.0))
        / (1.0 + np.sqrt(np.maximum(normalized, 0.0)))
    )
    exploratory *= 0.5 + 0.5 * (1.0 - rank_support)

    local_value = (
        0.50 * shortness
        + 0.20 * rank_support
        + 0.15 * endpoint_regret
        + 0.15 * bridge_potential
    )

    scores = local_value * (
        0.80
        + 0.15 * exploratory
        + 0.05 * (node_regret[:, None] + node_regret[None, :]) / 2.0
    )

    # Encourage symmetry and retain only valid off-diagonal edges.
    scores = 0.5 * (scores + scores.T)
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    minimum = np.min(scores)
    maximum = np.max(scores)
    if maximum > minimum:
        scores = (scores - minimum) / (maximum - minimum)
        scores[~np.isfinite(scores)] = 0.0
        np.fill_diagonal(scores, 0.0)

    return scores
