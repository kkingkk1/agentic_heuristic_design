import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a normalized symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & off_diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    safe = np.where(finite, np.maximum(distances, 0.0), np.inf)

    positive = safe[finite & (safe > 0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    row_counts = finite.sum(axis=1)
    row_values = np.where(finite, safe, np.nan)
    row_scale = np.nanmedian(row_values, axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        global_scale,
    )

    normalized = safe / global_scale
    local_normalized = safe / row_scale[:, None]

    inverse = np.zeros((n, n), dtype=float)
    inverse[finite] = 1.0 / (1.0 + normalized[finite])

    proximity = np.zeros((n, n), dtype=float)
    proximity[finite] = np.exp(-local_normalized[finite])

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.zeros((n, n), dtype=float)

    rank_values = np.zeros_like(safe, dtype=float)
    rank_values[finite] = 1.0 / np.sqrt(
        np.maximum(np.arange(1, n + 1, dtype=float)[None, :], 1.0)
    )

    for i in range(n):
        valid_order = order[i][finite[i, order[i]]]
        if valid_order.size:
            ranks[i, valid_order] = 1.0 / np.sqrt(
                np.arange(1, valid_order.size + 1, dtype=float)
            )

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(finite[i])
        if candidates.size:
            nearest[i] = candidates[np.argmin(safe[i, candidates])]

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    candidate_width = max(2, min(n - 1, int(np.ceil(np.sqrt(n)))))
    candidate = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = order[i][finite[i, order[i]]]
        if candidates.size:
            selected = candidates[:candidate_width]
            candidate[i, selected] = 1.0

    candidate = np.maximum(candidate, candidate.T)

    inverse_sym = 0.5 * (inverse + inverse.T)
    rank_sym = 0.5 * (ranks + ranks.T)
    proximity_sym = 0.5 * (proximity + proximity.T)
    mutual_sym = np.maximum(mutual, mutual.T)

    score = (
        0.40 * inverse_sym
        + 0.25 * rank_sym
        + 0.20 * proximity_sym
        + 0.10 * candidate
        + 0.05 * mutual_sym
    )

    valid_undirected = finite | finite.T
    score[~valid_undirected] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
