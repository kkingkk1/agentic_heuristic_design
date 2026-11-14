import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-prior indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(raw, dtype=float)

    positive_values = raw[valid & (raw > 0.0)]
    fallback = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    fallback = max(fallback, eps)

    d = np.where(valid, np.maximum(raw, 0.0), np.inf)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    global_scale = (
        float(np.median(finite_values))
        if finite_values.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    # Local distance scales based on each node's nearest valid neighbors.
    local_scale = np.empty(n, dtype=float)
    candidate_lists = []

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            local_scale[i] = global_scale
            candidate_lists.append(np.empty(0, dtype=int))
            continue

        ordered = neighbors[np.argsort(d[i, neighbors], kind="stable")]
        selected = ordered[:k]
        candidate_lists.append(selected)

        values = d[i, selected]
        local_scale[i] = max(float(np.median(values)), eps)

    pair_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )

    safe_d = np.where(valid, np.maximum(d, eps), np.inf)

    # Absolute and locally normalized distance attractiveness.
    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe_d[valid] + eps)
    inverse_max = np.max(inverse)
    if inverse_max > 0.0:
        inverse /= inverse_max

    local_affinity = np.zeros((n, n), dtype=float)
    local_affinity[valid] = np.exp(
        -safe_d[valid] / (pair_scale[valid] + eps)
    )

    # Two-sided reciprocal rank consensus.
    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        ordered = neighbors[np.argsort(d[i, neighbors], kind="stable")]
        rank_score[i, ordered] = 1.0 / (
            1.0 + np.arange(ordered.size, dtype=float)
        )

    rank_consensus = np.sqrt(rank_score * rank_score.T)

    # Mutual nearest-neighbor signal.
    nearest = np.full(n, -1, dtype=int)
    for i, neighbors in enumerate(candidate_lists):
        if neighbors.size:
            nearest[i] = neighbors[0]

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0

    # Smooth degree-pressure signal: preserve edges important to scarce nodes.
    normalized_inverse = np.zeros((n, n), dtype=float)
    row_sums = np.sum(inverse, axis=1, keepdims=True)
    np.divide(
        inverse,
        row_sums,
        out=normalized_inverse,
        where=row_sums > 0.0,
    )
    degree_pressure = np.sqrt(
        normalized_inverse * normalized_inverse.T
    )

    # Neighborhood bridge signal discourages excessive local subtours.
    neighborhood = np.zeros((n, n), dtype=float)
    for i, neighbors in enumerate(candidate_lists):
        neighborhood[i, neighbors] = 1.0

    overlap = neighborhood @ neighborhood.T
    bridge = 1.0 - overlap / max(float(k), 1.0)
    bridge = np.clip(bridge, 0.0, 1.0)
    bridge[~valid] = 0.0

    score = (
        0.30 * inverse
        + 0.28 * local_affinity
        + 0.20 * rank_consensus
        + 0.12 * mutual
        + 0.06 * degree_pressure
        + 0.04 * bridge
    )

    # Make the prior consistent for undirected TSP edges.
    score = 0.5 * (score + score.T)
    score[~valid & ~valid.T] = 0.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
