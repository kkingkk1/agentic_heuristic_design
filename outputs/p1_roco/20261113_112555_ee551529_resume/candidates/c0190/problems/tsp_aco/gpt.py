import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative symmetric edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    both_valid = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = finite.T & ~finite

    fused = np.full((n, n), np.inf, dtype=float)
    fused[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    fused[only_forward] = d[only_forward]
    fused[only_backward] = d.T[only_backward]
    np.fill_diagonal(fused, np.inf)

    valid = np.isfinite(fused)
    if not np.any(valid):
        return scores

    positive = fused[valid & (fused > 0.0)]
    eps = np.finfo(float).eps

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, global_scale, dtype=float)

    neighborhood = min(6, max(3, n - 1))

    for i in range(n):
        values = fused[i, np.isfinite(fused[i])]
        if values.size == 0:
            continue

        values = np.sort(values)
        nearest[i] = max(float(values[0]), 0.0)

        k = min(neighborhood, values.size)
        local_values = values[:k]

        if local_values.size >= 5:
            local_values = local_values[:-1]

        positive_local = local_values[local_values > 0.0]
        if positive_local.size:
            local_scale[i] = float(np.mean(positive_local))
        elif local_values.size:
            local_scale[i] = global_scale

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(np.maximum(
        np.outer(local_scale, local_scale),
        eps,
    ))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = fused / pair_scale
        shortness = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    masked = np.where(valid, fused, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")

    ranks = np.full((n, n), float(n), dtype=float)
    row_ids = np.arange(n)[:, None]
    positions = np.arange(n, dtype=float)[None, :]
    ordered_values = masked[row_ids, order]
    ordered_valid = np.isfinite(ordered_values)

    ranks[row_ids, order] = np.where(
        ordered_valid,
        positions,
        float(n),
    )

    rank_affinity = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(rank_affinity * rank_affinity.T)

    reciprocal = (
        (ranks < float(neighborhood))
        & (ranks.T < float(neighborhood))
    ).astype(float)

    # Weak future-potential term: prefer edges that are competitive at both ends,
    # while keeping its influence small and bounded.
    competitive = np.minimum(
        1.0,
        np.maximum(
            0.0,
            1.0 - ranks / max(float(neighborhood), 1.0),
        ),
    )
    future_potential = np.sqrt(competitive * competitive.T)

    scores = (
        shortness
        * (0.78 + 0.22 * rank_consensus)
        * (1.0 + 0.14 * reciprocal)
        * (0.94 + 0.06 * future_potential)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
