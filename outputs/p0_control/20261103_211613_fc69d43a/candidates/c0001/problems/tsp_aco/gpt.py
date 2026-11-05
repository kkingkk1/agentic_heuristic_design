import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability scores for a fixed ant-colony TSP solver."""
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

    # Robust global distance scale.
    finite_edges = work[np.isfinite(work)]
    positive_edges = finite_edges[finite_edges > 0]
    scale = np.median(positive_edges) if positive_edges.size else 1.0
    scale = max(float(scale), np.finfo(float).eps)

    # Local neighborhood structure.
    ordered = np.sort(work, axis=1)
    nearest = ordered[:, 0]
    second_nearest = ordered[:, 1] if n > 2 else nearest.copy()

    # Isolation: cities with larger nearest-neighbor distances receive more weight.
    isolation = nearest / (nearest + scale)
    isolation = 0.5 + isolation

    # Alternative-edge pressure: a large gap indicates that the edge may be
    # especially valuable relative to the endpoint's next-best alternative.
    local_gap = np.maximum(second_nearest - nearest, 0.0)
    local_gap = local_gap / (local_gap + scale)

    endpoint_value = isolation * (0.75 + 0.5 * local_gap)

    # Scale-free shortness, stabilized for duplicate or zero-length edges.
    normalized = work / scale
    shortness = 1.0 / (1.0 + normalized)

    # Edge-specific opportunity score: compare an edge with both endpoints'
    # local alternatives. This creates a scarcity-aware edge preference.
    endpoint_best = nearest[:, None] + nearest[None, :]
    edge_advantage = endpoint_best / (work + endpoint_best + scale)
    edge_advantage = np.where(np.isfinite(edge_advantage), edge_advantage, 0.0)

    scores = shortness * (
        0.55
        + 0.25 * edge_advantage
        + 0.20 * (endpoint_value[:, None] + endpoint_value[None, :]) / 2.0
    )

    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0
    return scores
