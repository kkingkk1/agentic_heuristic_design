import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive, symmetric edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    if not np.any(finite_nonnegative):
        return result

    positive_values = d[finite_nonnegative & (d > 0.0)]
    all_values = d[finite_nonnegative]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(all_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps

    # Convert possibly asymmetric input into a safe symmetric distance matrix.
    forward = np.where(finite_nonnegative, d, np.nan)
    backward = forward.T
    both = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = ~np.isfinite(forward) & np.isfinite(backward)

    safe = np.full((n, n), global_scale * 1e6, dtype=float)
    safe[both] = 0.5 * (forward[both] + backward[both])
    safe[only_forward] = forward[only_forward]
    safe[only_backward] = backward[only_backward]
    edge_valid = both | only_forward | only_backward
    np.fill_diagonal(safe, np.inf)
    np.fill_diagonal(edge_valid, False)

    valid_counts = edge_valid.sum(axis=1)
    if not np.any(edge_valid):
        return result

    # Estimate a robust local distance scale from up to four nearest valid edges.
    neighbor_count = min(4, n - 1)
    work = np.where(edge_valid, safe, np.inf)
    nearest_values = np.partition(work, kth=neighbor_count - 1, axis=1)
    local_values = nearest_values[:, :neighbor_count]

    local_scale = np.nanmedian(
        np.where(np.isfinite(local_values), local_values, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )

    # Sparse rows need a less aggressive scale to avoid suppressing all edges.
    local_scale = np.where(
        valid_counts > 0,
        np.maximum(local_scale, global_scale * 0.05),
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    ratio = np.divide(
        safe,
        pair_scale,
        out=np.full_like(safe, 50.0),
        where=np.isfinite(safe),
    )
    ratio = np.clip(ratio, 0.0, 50.0)

    # Adaptive shortness: preserves useful contrast while avoiding hard cutoffs.
    shortness = np.exp(-ratio)

    # Stable endpoint-relative ranks.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    rank_decay = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(rank_decay * rank_decay.T)

    # Soft reciprocal-neighbor bonus.
    nearest = np.argmin(work, axis=1)
    reciprocal = np.zeros((n, n), dtype=float)
    valid_nearest = np.isfinite(work[np.arange(n), nearest])

    for i in np.flatnonzero(valid_nearest):
        j = nearest[i]
        if nearest[j] == i:
            reciprocal[i, j] = 1.0
            reciprocal[j, i] = 1.0

    # Reward edges incident to nodes with few viable alternatives.
    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts.astype(float), 1.0))
    scarcity_pair = 0.5 * (scarcity[:, None] + scarcity[None, :])
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    # Mildly favor edges whose length is competitive within both endpoints.
    endpoint_competitiveness = np.sqrt(
        np.maximum(rank_decay * rank_decay.T, 0.0)
    )

    score = (
        shortness
        * (0.50 + 0.50 * rank_consensus)
        * (0.85 + 0.15 * endpoint_competitiveness)
        * (1.0 + 0.25 * reciprocal)
        * (0.90 + 0.30 * scarcity_pair)
    )

    score[~edge_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float)
