import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust symmetric edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    valid_both = finite & finite.T
    valid_forward = finite & ~finite.T
    valid_backward = finite.T & ~finite

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid_both] = 0.5 * (d[valid_both] + d.T[valid_both])
    symmetric[valid_forward] = d[valid_forward]
    symmetric[valid_backward] = d.T[valid_backward]
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return scores

    # Use a moderately broad, robust neighborhood for local normalization.
    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, global_scale, dtype=float)
    top_k = min(6, max(4, n - 1))

    for i in range(n):
        row = symmetric[i]
        values = row[np.isfinite(row)]
        if values.size == 0:
            continue

        values = np.sort(values)
        nearest[i] = values[0]
        k = min(top_k, values.size)
        local_scale[i] = float(np.mean(values[:k]))

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(
            np.outer(local_scale, local_scale),
            np.finfo(float).eps,
        )
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative = symmetric / pair_scale
        shortness = np.exp(-np.clip(relative, 0.0, 50.0))

    # Stable per-node ranking over valid neighbors only.
    masked = np.where(valid, symmetric, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=float)

    row_ids = np.arange(n)[:, None]
    valid_order = np.isfinite(masked[row_ids, order])
    ranks[row_ids, order] = np.where(
        valid_order,
        np.arange(n, dtype=float)[None, :],
        float(n),
    )

    # Smooth rank affinity; the geometric mean requires agreement at both ends.
    rank_affinity = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(rank_affinity * rank_affinity.T)

    # Reward reciprocal membership in a small top-k neighborhood, softly.
    reciprocal_topk = (
        (ranks < float(top_k))
        & (ranks.T < float(top_k))
    ).astype(float)

    # A limited endpoint-affinity term favors edges that are locally competitive,
    # without making isolated nearest-neighbor edges dominate.
    nearest_gap = np.maximum(local_scale, nearest)
    with np.errstate(divide="ignore", invalid="ignore"):
        endpoint_affinity = np.sqrt(
            np.clip(
                nearest_gap[:, None] / np.maximum(symmetric, np.finfo(float).eps),
                0.5,
                2.0,
            )
            * np.clip(
                nearest_gap[None, :] / np.maximum(symmetric, np.finfo(float).eps),
                0.5,
                2.0,
            )
        )

    scores = (
        shortness
        * (0.72 + 0.28 * rank_consensus)
        * (1.0 + 0.18 * reciprocal_topk)
        * (0.90 + 0.10 * endpoint_affinity)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
