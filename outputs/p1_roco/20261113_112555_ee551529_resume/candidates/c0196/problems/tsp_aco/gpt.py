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

    eps = np.finfo(float).eps
    positive = fused[valid & (fused > 0.0)]

    if positive.size:
        global_median = float(np.median(positive))
        global_q25 = float(np.percentile(positive, 25.0))
        global_scale = 0.7 * global_median + 0.3 * global_q25
    else:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    neighborhood = min(6, max(3, n - 1))
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = fused[i, np.isfinite(fused[i])]
        if row.size == 0:
            continue

        row = np.sort(row)
        nearest = row[: min(neighborhood, row.size)]
        positive_nearest = nearest[nearest > 0.0]

        if positive_nearest.size:
            local_mean = float(np.mean(positive_nearest))
            local_median = float(np.median(positive_nearest))
            local_scale[i] = 0.65 * local_mean + 0.35 * local_median

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(np.outer(local_scale, local_scale), eps)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        local_ratio = fused / pair_scale
        global_ratio = fused / global_scale

        # Truncation prevents a single exceptionally short edge from
        # overwhelming all other candidates.
        local_ratio = np.clip(local_ratio, 0.0, 12.0)
        global_ratio = np.clip(global_ratio, 0.0, 20.0)

        local_shortness = np.exp(-local_ratio)
        global_shortness = np.exp(-global_ratio)

    shortness = 0.88 * local_shortness + 0.12 * global_shortness

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

    rank_cap = max(float(neighborhood), 1.0)
    endpoint_quality = np.clip(1.0 - ranks / rank_cap, 0.0, 1.0)

    rank_consensus = np.sqrt(
        np.maximum(endpoint_quality * endpoint_quality.T, 0.0)
    )

    reciprocal = (
        (ranks < float(neighborhood))
        & (ranks.T < float(neighborhood))
    ).astype(float)

    # Mild competition factor: it rewards edges that are good at both
    # endpoints while keeping the correction deliberately bounded.
    endpoint_margin = np.maximum(
        0.0,
        1.0 - ranks / float(max(n - 1, 1)),
    )
    mutual_margin = np.sqrt(
        np.maximum(endpoint_margin * endpoint_margin.T, 0.0)
    )

    scores = (
        shortness
        * (0.84 + 0.16 * rank_consensus)
        * (1.0 + 0.10 * reciprocal)
        * (0.97 + 0.03 * mutual_margin)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
