import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return continuation-aware edge desirability scores for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite_positive & off_diagonal

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    valid_values = d[valid]
    scale = np.median(valid_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(valid_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_d = np.where(valid, d, np.inf)

    # Base preference for short edges, with a bounded tail for numerical stability.
    closeness = np.zeros_like(d, dtype=float)
    closeness[valid] = 1.0 / (1.0 + d[valid] / scale)

    # Destination continuation: an edge is stronger when its destination
    # has at least one good onward option.
    continuation = np.zeros_like(d, dtype=float)
    for j in range(n):
        outgoing = safe_d[j].copy()
        outgoing[j] = np.inf
        finite = np.isfinite(outgoing)
        if np.any(finite):
            best = np.min(outgoing[finite])
            continuation[:, j] = 1.0 / (1.0 + best / scale)

    # Regret/rarity: favor entering nodes whose good incoming choices are scarce.
    incoming_quality = np.zeros(n, dtype=float)
    for j in range(n):
        incoming = safe_d[:, j].copy()
        incoming[j] = np.inf
        finite = np.isfinite(incoming)
        if np.any(finite):
            ordered = np.sort(incoming[finite])
            k = min(3, ordered.size)
            incoming_quality[j] = np.mean(
                1.0 / (1.0 + ordered[:k] / scale)
            )

    rarity = np.zeros_like(incoming_quality)
    positive = incoming_quality > 0
    if np.any(positive):
        rarity[positive] = (
            1.0 - incoming_quality[positive] /
            np.max(incoming_quality[positive])
        )

    # Local competitiveness among each source's outgoing edges.
    competitiveness = np.zeros_like(d, dtype=float)
    for i in range(n):
        row = safe_d[i].copy()
        row[i] = np.inf
        finite = np.isfinite(row)
        if np.any(finite):
            local = np.zeros(n, dtype=float)
            local[finite] = 1.0 / (1.0 + row[finite] / scale)
            maximum = np.max(local)
            if maximum > 0:
                competitiveness[i] = local / maximum

    # Mutual-nearest-neighbor bonus, useful for preserving strong local structure.
    nearest_out = np.full(n, -1, dtype=int)
    nearest_in = np.full(n, -1, dtype=int)

    for i in range(n):
        row = safe_d[i].copy()
        row[i] = np.inf
        if np.any(np.isfinite(row)):
            nearest_out[i] = int(np.argmin(row))

    for j in range(n):
        column = safe_d[:, j].copy()
        column[j] = np.inf
        if np.any(np.isfinite(column)):
            nearest_in[j] = int(np.argmin(column))

    mutual = np.zeros_like(d, dtype=float)
    for i in range(n):
        j = nearest_out[i]
        if j >= 0 and nearest_in[j] == i:
            mutual[i, j] = 1.0

    scores = (
        0.52 * closeness
        + 0.20 * closeness * continuation
        + 0.13 * competitiveness
        + 0.10 * closeness * rarity[None, :]
        + 0.05 * mutual
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
