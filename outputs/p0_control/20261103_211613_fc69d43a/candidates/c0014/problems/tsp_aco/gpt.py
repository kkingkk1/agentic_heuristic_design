import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a positive edge-score matrix for a fixed TSP solver.

    Larger scores indicate edges that are preferred by the heuristic.
    Self-loops receive zero score.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D NumPy array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    # Local scale of each city: its nearest usable neighbor distance.
    nearest = np.min(work, axis=1)
    finite_nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

    if finite_nearest.size == 0:
        return scores

    fallback = float(np.median(finite_nearest))
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        fallback,
    )

    # Relative edge length: scale each edge by endpoint neighborhood density.
    local_scale = np.sqrt(nearest[:, None] * nearest[None, :])
    relative_length = work / np.maximum(local_scale, np.finfo(float).eps)

    # Local competition: rank each edge from both endpoints.
    order = np.argsort(work, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    competition = 1.0 / (1.0 + ranks + ranks.T)

    # Two-hop bridge quality through the best common intermediary.
    two_hop = work[:, :, None] + work[None, :, :]
    bridge = np.min(two_hop, axis=1)
    bridge_ratio = bridge / np.maximum(work, np.finfo(float).eps)

    # Strong affinity for short, locally competitive edges with good continuity.
    affinity = (
        np.exp(-relative_length)
        * (0.5 + competition)
        * (1.0 + 0.35 / (1.0 + bridge_ratio))
    )

    # Reciprocal form keeps scores positive and limits extreme scale effects.
    score = affinity / np.maximum(work, np.finfo(float).eps)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    return score
