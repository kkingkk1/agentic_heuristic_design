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

    # Safely convert asymmetric input into a symmetric distance matrix.
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

    if not np.any(edge_valid):
        return result

    valid_counts = edge_valid.sum(axis=1)

    # Estimate local scales from up to five nearest valid edges.
    neighbor_count = min(5, n - 1)
    work = np.where(edge_valid, safe, np.inf)

    nearest_values = np.partition(
        work,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    finite_nearest = np.where(
        np.isfinite(nearest_values),
        nearest_values,
        np.nan,
    )

    local_scale = np.nanmedian(finite_nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )

    # Prevent sparse rows from receiving excessively sharp penalties.
    local_scale = np.where(
        valid_counts > 0,
        np.maximum(local_scale, global_scale * 0.08),
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    ratio = np.divide(
        safe,
        pair_scale,
        out=np.full_like(safe, 60.0),
        where=np.isfinite(safe),
    )
    ratio = np.clip(ratio, 0.0, 60.0)

    # Smooth adaptive shortness with stronger contrast for competitive edges.
    shortness = np.exp(-0.85 * ratio)

    # Stable endpoint-relative ranks.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    rank_decay = 1.0 / np.power(1.0 + ranks, 0.85)

    # Geometric mean emphasizes edges preferred by both endpoints.
    rank_consensus = np.sqrt(
        np.maximum(rank_decay * rank_decay.T, 0.0)
    )

    # Soft reciprocal-neighbor bonus.
    nearest = np.argmin(work, axis=1)
    reciprocal = np.zeros((n, n), dtype=float)
    nearest_is_valid = np.isfinite(work[np.arange(n), nearest])

    for i in np.flatnonzero(nearest_is_valid):
        j = nearest[i]
        if nearest[j] == i:
            reciprocal[i, j] = 1.0
            reciprocal[j, i] = 1.0

    # Reward edges connected to nodes with fewer viable alternatives.
    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts.astype(float), 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    endpoint_competitiveness = np.sqrt(
        np.maximum(rank_decay * rank_decay.T, 0.0)
    )

    score = (
        shortness
        * (0.40 + 0.60 * rank_consensus)
        * (0.80 + 0.20 * endpoint_competitiveness)
        * (1.0 + 0.35 * reciprocal)
        * (0.88 + 0.32 * scarcity_pair)
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
