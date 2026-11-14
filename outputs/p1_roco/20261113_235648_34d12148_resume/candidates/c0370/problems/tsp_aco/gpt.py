import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, direction-preserving edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = offdiag & finite

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = raw[valid & (raw > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    cleaned = np.zeros((n, n), dtype=float)
    cleaned[valid] = np.maximum(raw[valid], 0.0)

    observed = cleaned[valid]
    maximum = float(np.max(observed)) if observed.size else scale
    penalty = maximum + 16.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps

    # Strong local preference for short edges.
    temperature = max(0.42 * scale, eps)
    affinity = np.exp(-np.minimum(distances / temperature, 60.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    # Direction-preserving local ranks.
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for row in range(n):
        ranks[row, order[row]] = rank_values

    local_width = max(2.0, min(8.0, np.sqrt(float(n)) + 1.0))
    rank_affinity = np.exp(-ranks / local_width)
    rank_affinity[~valid] = 0.0
    np.fill_diagonal(rank_affinity, 0.0)

    # Normalize distance evidence per source without erasing absolute differences.
    row_max = np.maximum(np.max(affinity, axis=1, keepdims=True), eps)
    local_affinity = affinity / row_max

    # Mild rarity signal: favor underrepresented destinations only when
    # the edge is already locally plausible.
    outgoing = affinity / np.maximum(
        affinity.sum(axis=1, keepdims=True), eps
    )
    incoming = affinity / np.maximum(
        affinity.sum(axis=0, keepdims=True), eps
    )

    support = np.sqrt(np.maximum(outgoing * incoming, 0.0))
    rarity = 1.0 - np.sqrt(np.clip(support, 0.0, 1.0))

    plausibility = np.sqrt(
        np.clip(local_affinity, 0.0, 1.0)
        * np.clip(rank_affinity, 0.0, 1.0)
    )
    diversity = np.minimum(0.35, rarity) * plausibility

    # Preserve a small exploration floor for every valid edge.
    floor = 0.012 * (0.35 + 0.65 * rank_affinity)

    score = (
        0.72 * local_affinity
        + 0.20 * rank_affinity
        + 0.06 * diversity
        + floor
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
